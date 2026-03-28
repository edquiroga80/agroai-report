# agro_charts.py — Gráficos y cultivos
# GIS Dev Academy — AgroIA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json
import importlib, inspect, pkgutil
from datetime import date


MESES_N = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


# ── Cultivos por defecto ───────────────────────────────────────────────────
def cultivos_default(lat):
    sur = lat < -35
    if sur:
        return [
            {"cultivo":"Trigo",     "tipo":"invierno","sd_i":152,"sd_f":212,"cd_i":305,"cd_f":360,
             "color":"#639922","sh":"media","sg":"alta","fch":"Espigado oct","fcg":"Grano lechoso nov"},
            {"cultivo":"Cebada",    "tipo":"invierno","sd_i":135,"sd_f":182,"cd_i":285,"cd_f":335,
             "color":"#97C459","sh":"media","sg":"alta","fch":"Encañado jul-ago","fcg":"Espigazón oct"},
            {"cultivo":"Girasol",   "tipo":"verano",  "sd_i":274,"sd_f":335,"cd_i":50, "cd_f":90,
             "color":"#BA7517","sh":"alta","sg":"media","fch":"Emergencia oct","fcg":"Antesis dic-ene"},
            {"cultivo":"Soja",      "tipo":"verano",  "sd_i":305,"sd_f":365,"cd_i":60, "cd_f":120,
             "color":"#1D9E75","sh":"alta","sg":"media","fch":"Emergencia nov","fcg":"Llenado grano feb"},
            {"cultivo":"Maíz",      "tipo":"verano",  "sd_i":274,"sd_f":320,"cd_i":60, "cd_f":121,
             "color":"#EF9F27","sh":"alta","sg":"alta","fch":"Emergencia oct","fcg":"Floración dic-ene"},
            {"cultivo":"Ganadería", "tipo":"perenne", "sd_i":1,  "sd_f":365,"cd_i":1,  "cd_f":365,
             "color":"#888780","sh":"baja","sg":"baja","fch":"Pastura invierno","fcg":"Implantación"},
        ]
    else:
        return [
            {"cultivo":"Soja",      "tipo":"verano",  "sd_i":305,"sd_f":365,"cd_i":60, "cd_f":120,
             "color":"#1D9E75","sh":"alta","sg":"media","fch":"Emergencia oct-nov","fcg":"R5-R6 feb-mar"},
            {"cultivo":"Maíz",      "tipo":"verano",  "sd_i":274,"sd_f":335,"cd_i":50, "cd_f":121,
             "color":"#EF9F27","sh":"alta","sg":"alta","fch":"Emergencia","fcg":"Floración dic-ene"},
            {"cultivo":"Trigo",     "tipo":"invierno","sd_i":152,"sd_f":212,"cd_i":305,"cd_f":365,
             "color":"#639922","sh":"media","sg":"alta","fch":"Espigado oct","fcg":"Grano lechoso nov"},
            {"cultivo":"Girasol",   "tipo":"verano",  "sd_i":274,"sd_f":320,"cd_i":50, "cd_f":90,
             "color":"#BA7517","sh":"media","sg":"media","fch":"Emergencia","fcg":"Antesis dic"},
            {"cultivo":"Soja 2da",  "tipo":"verano",  "sd_i":335,"sd_f":365,"cd_i":90, "cd_f":150,
             "color":"#5DCAA5","sh":"alta","sg":"media","fch":"Emergencia dic","fcg":"R5-R6 mar"},
            {"cultivo":"Ganadería", "tipo":"perenne", "sd_i":1,  "sd_f":365,"cd_i":1,  "cd_f":365,
             "color":"#888780","sh":"baja","sg":"baja","fch":"Pastura invierno","fcg":"Implantación"},
        ]


def estimar_cultivos_mistral(lat, lon, nombre, api_key):
    """Consulta Mistral AI para estimar cultivos de la zona."""
    try:
        import mistralai
        candidatos = []
        for _, modname, _ in pkgutil.walk_packages(
                path=mistralai.__path__, prefix="mistralai.", onerror=lambda x: None):
            try:
                mod = importlib.import_module(modname)
                for n, obj in inspect.getmembers(mod, inspect.isclass):
                    if "client" in n.lower() or n == "Mistral":
                        candidatos.append((n, obj, modname))
            except Exception:
                pass
        cls = None
        for n, c, _ in candidatos:
            if n == "Mistral": cls = c; break
        if cls is None:
            for n, c, _ in candidatos:
                if "mistral" in n.lower(): cls = c; break
        if cls is None:
            return cultivos_default(lat)

        client = cls(api_key=api_key)
        prompt = (
            f"Sos agrónomo experto en Argentina. Para lat={lat:.3f}, lon={lon:.3f} ({nombre}), "
            f"listá los 6 cultivos más representativos.\n"
            f"Respondé SOLO con JSON válido (sin markdown) con esta estructura:\n"
            f'[{{"cultivo":"Trigo","tipo":"invierno","sd_i":152,"sd_f":212,"cd_i":305,"cd_f":360,'
            f'"color":"#639922","sh":"media","sg":"alta","fch":"Espigado oct","fcg":"Grano lechoso"}}]\n'
            f"sd_i/sd_f=día inicio/fin siembra (1-365), cd_i/cd_f=día cosecha. "
            f"sh=sensibilidad helada (alta|media|baja), sg=sensibilidad granizo."
        )
        try:
            r = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=1200)
        except AttributeError:
            r = client.chat(
                model="mistral-large-latest",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=1200)
        raw = r.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        cultivos = json.loads(raw)
        for c in cultivos:
            for k in ["sd_i","sd_f","cd_i","cd_f"]:
                c[k] = int(c.get(k, 1))
        return cultivos
    except Exception:
        return cultivos_default(lat)


def _barra(ax, y, xi, xf, color, alpha, h, label=None, lw=0.4):
    if xi <= xf:
        ancho = xf - xi
        if ancho > 0:
            ax.barh(y, ancho, left=xi, height=h, color=color, alpha=alpha,
                    edgecolor="white", linewidth=lw, zorder=3)
            if label and ancho > 14:
                ax.text(xi+ancho/2, y, label, ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=4)
    else:
        ax.barh(y, 365-xi, left=xi, height=h, color=color, alpha=alpha,
                edgecolor="white", linewidth=lw, zorder=3)
        ax.barh(y, xf, left=1, height=h, color=color, alpha=alpha,
                edgecolor="white", linewidth=lw, zorder=3)
        if label:
            ax.text(min(xi+(365-xi)/2,363), y, label, ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)


def generar_grafico(cal, cultivos, nombre, lat, lon):
    """Genera el gráfico de 4 paneles y devuelve la figura matplotlib."""
    BG = "#F8F7F4"; GRID = "#E8E7E3"; TICK = "#888780"; TEXT = "#333330"

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4,1, height_ratios=[4.5,1.2,1.2,1.2],
                            hspace=0.42, left=0.10, right=0.97, top=0.94, bottom=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    MTICKS  = [1,32,60,91,121,152,182,213,244,274,305,335]
    MLABELS = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    def setup(ax, title="", ylabel=""):
        ax.set_xlim(0,366); ax.set_facecolor(BG)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(GRID)
        ax.tick_params(colors=TICK, labelsize=9)
        ax.set_xticks(MTICKS); ax.set_xticklabels(MLABELS, color=TICK)
        for m in MTICKS: ax.axvline(m, color=GRID, lw=0.6, zorder=0)
        if title: ax.set_title(title, loc="left", fontsize=9.5, color=TICK, pad=4)
        if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=TICK, labelpad=5)

    ax0,ax1,ax2,ax3 = axes
    fig.suptitle(f"Calendario Agroclimático  ·  {nombre}  (lat={lat:.3f}, lon={lon:.3f})",
                 fontsize=13, color=TEXT, x=0.53)
    setup(ax0, ylabel="Cultivo")

    N = len(cultivos); BH = 0.52
    for _, row in cal.iterrows():
        d = row["dia"]
        if row["clase"]==1:
            ax0.axvspan(d-.5,d+.5,color="#EF9F27",alpha=0.11,zorder=0)
        elif row["clase"]>=2:
            ax0.axvspan(d-.5,d+.5,color="#E24B4A",alpha=0.14,zorder=0)
        if row["p_gran"]>0.06:
            ax0.axvspan(d-.5,d+.5,color="#185FA5",alpha=float(row["p_gran"])*.5,zorder=0)
        if row["cls_sequia"]>=2:
            ax0.axvspan(d-.5,d+.5,color="#854F0B",alpha=0.13,zorder=0)
        elif row["cls_sequia"]==1:
            ax0.axvspan(d-.5,d+.5,color="#EF9F27",alpha=0.07,zorder=0)

    for i, c in enumerate(cultivos):
        y = N-1-i; col = c["color"]; tipo = c["tipo"]
        if tipo=="perenne":
            ax0.barh(y,365,left=1,height=BH,color=col,alpha=0.40,
                     edgecolor="white",linewidth=0.4,zorder=3)
            ax0.text(183,y,c["cultivo"],ha="center",va="center",
                     fontsize=8,color="white",fontweight="bold",zorder=4)
        else:
            _barra(ax0,y,c["sd_i"],c["sd_f"],col,0.90,BH,"Siembra")
            _barra(ax0,y,c["cd_i"],c["cd_f"],col,0.90,BH,"Cosecha")
            if c["sd_f"]<c["cd_i"]:
                _barra(ax0,y,c["sd_f"],c["cd_i"],col,0.28,BH*.65)
            elif c["sd_f"]!=c["cd_i"]:
                _barra(ax0,y,c["sd_f"],365,col,0.28,BH*.65)
                _barra(ax0,y,1,c["cd_i"],col,0.28,BH*.65)
        if c.get("sh","baja")=="alta":
            da = int((cal["clase"]>=2).sum())
            if da>0:
                pico = int(cal[cal["clase"]>=2]["dia"].iloc[da//2])
                ax0.plot(pico,y+BH/2+0.08,marker="v",color="#E24B4A",ms=6,zorder=5)

    ax0.set_yticks(range(N))
    ax0.set_yticklabels([c["cultivo"] for c in reversed(cultivos)],
                         fontsize=10.5,color=TEXT)
    ax0.set_ylim(-.5,N-.5)
    ax0.tick_params(axis="y",length=0,pad=6)

    dia_hoy = date.today().timetuple().tm_yday
    for ax in axes:
        ax.axvline(dia_hoy,color="#E24B4A",lw=1.4,ls="--",alpha=0.65,zorder=6)
    ax0.text(dia_hoy+1.5,N-.15,"Hoy",fontsize=8.5,color="#E24B4A",zorder=7)

    leyenda = [
        mpatches.Patch(color="#EF9F27",alpha=0.55,label="Helada leve (3-5°C)"),
        mpatches.Patch(color="#E24B4A",alpha=0.65,label="Helada alta (≤3°C)"),
        mpatches.Patch(color="#185FA5",alpha=0.55,label="Granizo"),
        mpatches.Patch(color="#854F0B",alpha=0.55,label="Sequía moderada+"),
        plt.Line2D([0],[0],color="#E24B4A",lw=1.4,ls="--",label="Hoy"),
        plt.Line2D([0],[0],color="#E24B4A",marker="v",ms=6,ls="",label="Fase crítica helada"),
    ]
    ax0.legend(handles=leyenda,fontsize=7.5,loc="upper right",
               framealpha=0.88,ncol=3,borderpad=0.6)

    # Panel heladas
    setup(ax1,"Riesgo de heladas (climatología histórica)","Prob.")
    ax1.fill_between(cal["dia"],cal["p_alto"],color="#E24B4A",alpha=0.72,label="Riesgo alto/helada")
    ax1.fill_between(cal["dia"],cal["p_leve"],color="#EF9F27",alpha=0.55,label="Riesgo leve")
    ax1b = ax1.twinx()
    ax1b.plot(cal["dia"],cal["tmin_clim"],color="#185FA5",lw=1.0,alpha=0.5,linestyle=":")
    ax1b.set_ylabel("Tmin (°C)",fontsize=8,color="#185FA5")
    ax1b.tick_params(colors="#185FA5",labelsize=8)
    ax1b.spines[["top"]].set_visible(False)
    ax1.set_ylim(0,1); ax1.set_xlim(0,366)
    ax1.legend(fontsize=7.5,loc="upper left",framealpha=0.8,ncol=2)

    # Panel granizo
    setup(ax2,"Riesgo de granizo (climatología histórica)","Prob.")
    ax2.fill_between(range(1,366),cal["p_gran"],color="#185FA5",alpha=0.65,label="Granizo")
    ax2.set_ylim(0,max(0.22,float(cal["p_gran"].max())*1.3))
    ax2.set_xlim(0,366)
    ax2.legend(fontsize=7.5,loc="upper left",framealpha=0.8)

    # Panel sequía
    setup(ax3,"Índice de sequía SPI-3 (climatología histórica)","Prob.")
    ax3.fill_between(cal["dia"],cal["p_seq_sev"],color="#712B13",alpha=0.85,label="Severa/extrema")
    ax3.fill_between(cal["dia"],cal["p_seq_mod"],color="#D85A30",alpha=0.65,label="Moderada")
    ax3.fill_between(cal["dia"],cal["p_seq_leve"],color="#EF9F27",alpha=0.45,label="Leve")
    ax3b = ax3.twinx()
    ax3b.plot(cal["dia"],cal["spi_clim"],color="#854F0B",lw=1.1,alpha=0.65,linestyle="-.")
    for umbral,color in [(-0.5,"#EF9F27"),(-1.0,"#D85A30"),(-1.5,"#712B13")]:
        ax3b.axhline(umbral,color=color,lw=0.6,ls=":",alpha=0.7)
    ax3b.set_ylabel("SPI-3",fontsize=8,color="#854F0B")
    ax3b.tick_params(colors="#854F0B",labelsize=8)
    ax3b.spines[["top"]].set_visible(False)
    ax3.set_ylim(0,1); ax3.set_xlim(0,366)
    ax3.legend(fontsize=7.5,loc="upper left",framealpha=0.8,ncol=3)

    plt.tight_layout(rect=[0,0,1,0.97])
    return fig
