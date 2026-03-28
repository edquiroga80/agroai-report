# agro_engine.py — Motor de datos, modelos ML e índices climáticos
# GIS Dev Academy — AgroIA

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ── Variables Open-Meteo ───────────────────────────────────────────────────
VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_min",
    "dewpoint_2m_max", "dewpoint_2m_min", "dewpoint_2m_mean",
    "relative_humidity_2m_max", "relative_humidity_2m_min",
    "precipitation_sum", "windspeed_10m_max", "windgusts_10m_max",
    "cloudcover_mean", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "weather_code",
]
COLS = [
    "tmax","tmin","tmean","tapp_min","rocio_max","rocio_min","rocio_mean",
    "hr_max","hr_min","lluvia","viento_max","racha_max","nubosidad_mean",
    "radiacion_sum","etp","weather_code",
]
FEAT_H = [
    "tmax","tmean","tapp_min","rocio_min","rocio_mean","hr_max","hr_min",
    "dep_rocio","amplitud","delta_tmin","delta_tmax",
    "tmin_3d","tmin_7d","anomalia_tmin",
    "lluvia","lluvia_3d","viento_max","racha_max",
    "nubosidad_mean","radiacion_sum","cielo_desp","etp",
    "dia_sin","dia_cos","mes_sin","mes_cos",
]
FEAT_G = [
    "tmax","tmean","amplitud","amplitud_3d",
    "lluvia","lluvia_3d","lluvia_7d",
    "racha_max","racha_3d","viento_max",
    "nubosidad_mean","hr_max","radiacion_sum",
    "dia_sin","dia_cos","mes_sin","mes_cos",
]
ETQ_H = {0:"Sin riesgo", 1:"Riesgo leve", 2:"Riesgo alto/helada"}
ETQ_S = {0:"Normal", 1:"Sequía leve", 2:"Sequía moderada",
         3:"Sequía severa", 4:"Sequía extrema"}
MESES = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


def _om_client():
    cache = requests_cache.CachedSession(".cache_agro", expire_after=-1)
    return openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))


def descargar_historico(lat, lon, anios=10, progress_cb=None):
    """Descarga datos ERA5 históricos desde Open-Meteo."""
    hoy = date.today()
    fecha_fin = hoy.replace(month=1, day=1).isoformat()
    fecha_ini = str(int(fecha_fin[:4]) - anios) + fecha_fin[4:]
    if progress_cb:
        progress_cb(f"Descargando {anios} años ERA5 ({fecha_ini} → {fecha_fin})...")
    om = _om_client()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": fecha_ini, "end_date": fecha_fin,
        "daily": VARIABLES,
        "timezone": "America/Argentina/Buenos_Aires",
        "models": "era5_seamless",
    }
    resp = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
    daily = resp.Daily()
    df = pd.DataFrame(
        {c: daily.Variables(i).ValuesAsNumpy() for i, c in enumerate(COLS)},
        index=pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True).tz_localize(None),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_localize(None),
            freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left")
    )
    df.index.name = "fecha"
    return df.dropna(subset=["tmax","tmin"])


def _feat_engineering(df):
    """Calcula los 26 features agroclimáticos."""
    d = df.copy()
    d["dep_rocio"]     = d["tmin"] - d["rocio_min"]
    d["amplitud"]      = d["tmax"] - d["tmin"]
    d["delta_tmin"]    = d["tmin"].diff()
    d["delta_tmax"]    = d["tmax"].diff()
    d["tmin_3d"]       = d["tmin"].rolling(3).mean()
    d["tmin_7d"]       = d["tmin"].rolling(7).mean()
    d["tmin_30d"]      = d["tmin"].rolling(30).mean()
    d["anomalia_tmin"] = d["tmin"] - d["tmin_30d"]
    d["mes"]           = d.index.month
    d["dia_anio"]      = d.index.dayofyear
    d["dia_sin"]       = np.sin(2*np.pi*d["dia_anio"]/365)
    d["dia_cos"]       = np.cos(2*np.pi*d["dia_anio"]/365)
    d["mes_sin"]       = np.sin(2*np.pi*d["mes"]/12)
    d["mes_cos"]       = np.cos(2*np.pi*d["mes"]/12)
    d["cielo_desp"]    = (d["nubosidad_mean"] < 20).astype(int)
    d["lluvia_3d"]     = d["lluvia"].rolling(3).sum()
    d["lluvia_7d"]     = d["lluvia"].rolling(7).sum()
    d["racha_3d"]      = d["racha_max"].rolling(3).max()
    d["amplitud_3d"]   = d["amplitud"].rolling(3).mean()
    # Etiquetas
    d["cls_helada"] = d["tmin"].apply(
        lambda t: 2 if t <= 3.0 else (1 if t <= 5.0 else 0))
    GRAN = {89,90,96,99}
    d["granizo"] = d["weather_code"].apply(
        lambda c: 1 if int(c) in GRAN else 0)
    return d.dropna()


def entrenar_modelo_heladas(df, modelo_path="modelo_heladas.pkl", progress_cb=None):
    """Entrena XGBoost para heladas o carga existente."""
    if os.path.exists(modelo_path):
        if progress_cb: progress_cb("Cargando modelo de heladas existente...")
        b = joblib.load(modelo_path)
        return b["modelo"], b["scaler"]

    if progress_cb: progress_cb("Entrenando modelo de heladas XGBoost...")
    X = df[FEAT_H].values; y = df["cls_helada"].values
    cnt = {c: int((y==c).sum()) for c in np.unique(y)}
    estrat = all(v >= 2 for v in cnt.values())
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                        stratify=y if estrat else None)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    pm = {c: len(ytr)/(len(cnt)*n) for c,n in cnt.items() if c in np.unique(ytr)}
    sw = np.array([pm.get(yi, 1.0) for yi in ytr])
    modelo = XGBClassifier(objective="multi:softprob", num_class=3,
                            n_estimators=200, max_depth=5, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=0)
    modelo.fit(Xtr_s, ytr, sample_weight=sw,
               eval_set=[(Xte_s, yte)], verbose=False)
    f1 = f1_score(yte, modelo.predict(Xte_s), average="weighted")
    joblib.dump({"modelo":modelo,"scaler":scaler,"features":FEAT_H,"etiquetas":ETQ_H},
                modelo_path)
    if progress_cb: progress_cb(f"Modelo heladas guardado (F1={f1:.3f})")
    return modelo, scaler


def modelo_granizo(df, progress_cb=None):
    """Entrena Random Forest para granizo o usa modelo paramétrico."""
    n_gran = int(df["granizo"].sum())
    if n_gran >= 20:
        if progress_cb: progress_cb(f"Entrenando modelo granizo ({n_gran} eventos ERA5)...")
        rng = np.random.default_rng(42)
        idx_p = np.where(df["granizo"].values==1)[0]
        idx_n = rng.choice(np.where(df["granizo"].values==0)[0],
                           min(len(idx_p)*10, (df["granizo"]==0).sum()), replace=False)
        idx_b = np.concatenate([idx_p, idx_n]); rng.shuffle(idx_b)
        Xb = df[FEAT_G].values[idx_b]; yb = df["granizo"].values[idx_b]
        Xtr,Xte,ytr,yte = train_test_split(Xb, yb, test_size=0.2,
                                            random_state=42, stratify=yb)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        m = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(Xtr_s, ytr)
        clim_g = df.groupby("dia_anio")[FEAT_G].mean().reindex(
            pd.RangeIndex(1,366)).interpolate()
        return sc.transform(clim_g[FEAT_G].values), m, True
    else:
        if progress_cb: progress_cb("Usando modelo estacional de granizo (ERA5 insuficiente)...")
        dias = np.arange(1,366)
        prob_raw = 0.5*(1+np.cos(2*np.pi*(dias-290)/365))
        prob = 0.015 + 0.16*prob_raw
        inv = (dias>=152)&(dias<=244)
        prob[inv] *= 0.10
        return prob, None, False


def calendario_sequia(df, progress_cb=None):
    """Calcula SPI-3 y probabilidades de sequía por día del año."""
    if progress_cb: progress_cb("Calculando SPI-3 y balance hídrico...")
    d = df.copy()
    d["bal_hidrico"] = d["lluvia"] - d["etp"]
    d["bal_30d"]     = d["bal_hidrico"].rolling(30, min_periods=15).sum()
    d["bal_90d"]     = d["bal_hidrico"].rolling(90, min_periods=45).sum()
    d["lluvia_30d"]  = d["lluvia"].rolling(30, min_periods=15).sum()
    d["lluvia_90d"]  = d["lluvia"].rolling(90, min_periods=45).sum()
    d["etp_30d"]     = d["etp"].rolling(30, min_periods=15).sum()
    d["dia_anio"]    = d.index.dayofyear
    d = d.dropna(subset=["bal_90d","lluvia_90d"])

    stats = d.groupby("dia_anio")["lluvia_90d"].agg(["mean","std"]).reset_index()
    stats.columns = ["dia_anio","ll90_mean","ll90_std"]
    stats["ll90_std"] = stats["ll90_std"].clip(lower=1.0)

    idx_orig = d.index
    d = d.merge(stats, on="dia_anio", how="left")
    d.index = idx_orig
    d["spi3"] = (d["lluvia_90d"] - d["ll90_mean"]) / d["ll90_std"]
    d["cls_sequia"] = d["spi3"].apply(
        lambda s: 0 if s>=-0.5 else (1 if s>=-1.0 else (2 if s>=-1.5 else (3 if s>=-2.0 else 4))))

    d["mes"]     = d.index.month
    d["dia_sin"] = np.sin(2*np.pi*d["dia_anio"]/365)
    d["dia_cos"] = np.cos(2*np.pi*d["dia_anio"]/365)
    d["mes_sin"] = np.sin(2*np.pi*d["mes"]/12)
    d["mes_cos"] = np.cos(2*np.pi*d["mes"]/12)

    IDX = pd.RangeIndex(1,366)
    prob_seq = np.zeros((365,5))
    for dia in IDX:
        sub = d[d["dia_anio"]==dia]["cls_sequia"]
        if len(sub) > 0:
            for cls in range(5):
                prob_seq[dia-1,cls] = (sub==cls).mean()
        else:
            prob_seq[dia-1,0] = 1.0

    spi_clim = d.groupby("dia_anio")["spi3"].mean().reindex(IDX).interpolate()
    bal_clim = d.groupby("dia_anio")["bal_90d"].mean().reindex(IDX).interpolate()
    return prob_seq, spi_clim, bal_clim


def generar_calendario(df_raw, modelo_h, scaler_h, gran_data, gran_model, gran_ml,
                        prob_seq, spi_clim, bal_clim, lat, progress_cb=None):
    """Genera el DataFrame calendario de 365 días con todas las amenazas."""
    if progress_cb: progress_cb("Generando calendario de 365 días...")
    df = _feat_engineering(df_raw)
    df["dia_anio"] = df.index.dayofyear

    IDX = pd.RangeIndex(1,366)
    fechas_ref = pd.date_range("2024-01-01", periods=365)

    # Heladas
    clim_h = df.groupby("dia_anio")[FEAT_H].mean().reindex(IDX).interpolate()
    X_cal  = scaler_h.transform(clim_h[FEAT_H].values)
    cls_h  = modelo_h.predict(X_cal)
    prb_h  = modelo_h.predict_proba(X_cal)

    # Granizo
    if gran_ml and gran_model is not None:
        scaler_g, modelo_g = gran_data
        prob_gran = modelo_g.predict_proba(gran_data)[:,1]
    else:
        prob_gran = gran_data if isinstance(gran_data, np.ndarray) else gran_data[:365]

    cal = pd.DataFrame({
        "dia":       range(1,366),
        "mes":       fechas_ref.month,
        "tmin_clim": df.groupby("dia_anio")["tmin"].mean().reindex(IDX).interpolate().values.round(2),
        "clase":     cls_h,
        "etiqueta":  [ETQ_H[c] for c in cls_h],
        "p_sin":     prb_h[:,0].round(4),
        "p_leve":    prb_h[:,1].round(4),
        "p_alto":    prb_h[:,2].round(4),
        "p_gran":    prob_gran[:365],
        "spi_clim":  spi_clim.values,
        "bal_clim":  bal_clim.values,
        "cls_sequia": np.argmax(prob_seq, axis=1),
        "etq_sequia": [ETQ_S[c] for c in np.argmax(prob_seq, axis=1)],
        "p_seq_leve": prob_seq[:,1],
        "p_seq_mod":  prob_seq[:,2],
        "p_seq_sev":  prob_seq[:,3]+prob_seq[:,4],
        "p_sequia":   1-prob_seq[:,0],
    })
    return cal


def pipeline_completo(lat, lon, nombre, modelo_path="modelo_heladas.pkl",
                       anios=10, progress_cb=None):
    """Ejecuta el pipeline completo y devuelve el calendario."""
    df_raw = descargar_historico(lat, lon, anios, progress_cb)
    df     = _feat_engineering(df_raw)

    modelo_h, scaler_h = entrenar_modelo_heladas(df, modelo_path, progress_cb)

    gran_data_raw, gran_model, gran_ml = modelo_granizo(df, progress_cb)

    prob_seq, spi_clim, bal_clim = calendario_sequia(df_raw, progress_cb)

    cal = generar_calendario(df_raw, modelo_h, scaler_h,
                              gran_data_raw, gran_model, gran_ml,
                              prob_seq, spi_clim, bal_clim, lat, progress_cb)
    return cal, df_raw
