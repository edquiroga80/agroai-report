# app.py — AgroIA Report · Streamlit App
# GIS Dev Academy — gisdev.ar
# Corré con: streamlit run app.py

import streamlit as st
import folium
from streamlit_folium import st_folium
import tempfile, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agro_engine import pipeline_completo
from agro_charts  import estimar_cultivos_mistral, cultivos_default, generar_grafico
from agro_report  import generar_texto_mistral, generar_pdf

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroIA Report · GIS Dev Academy",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS mínimo ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.stButton>button{width:100%;background:#1D9E75;color:white;border:none;
    border-radius:8px;padding:.6rem;font-weight:500;font-size:15px}
.stButton>button:hover{background:#0F6E56}
.metric-card{background:#F8F7F4;border:1px solid #E8E7E3;border-radius:10px;
    padding:14px 16px;text-align:center}
.metric-val{font-size:28px;font-weight:500;color:#1D9E75;line-height:1.1}
.metric-lbl{font-size:11px;color:#888780;margin-top:3px}
.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:500}
.badge-ok{background:#E1F5EE;color:#085041}
.badge-warn{background:#FAEEDA;color:#633806}
.badge-danger{background:#FCEBEB;color:#791F1F}
.badge-blue{background:#E6F1FB;color:#0C447C}
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ───────────────────────────────────────────────────────
for k, v in {
    "cal": None, "cultivos": None, "fig": None,
    "texto": None, "pdf_bytes": None,
    "lat": -37.47, "lon": -61.87, "nombre": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://placehold.co/280x60/1D9E75/white?text=GIS+Dev+Academy",
             use_container_width=True)
    st.markdown("### AgroIA Report")
    st.markdown("Calendario agroclimático anual con IA para cualquier campo argentino.")
    st.divider()

    st.markdown("**1. Datos del campo**")
    nombre = st.text_input("Nombre del campo", value="Mi Campo",
                            placeholder="Ej: La Esperanza")
    st.markdown("**2. Ubicación** — hacé clic en el mapa o ingresá las coordenadas")
    col1, col2 = st.columns(2)
    with col1:
        lat_input = st.number_input("Latitud", value=st.session_state.lat,
                                     format="%.5f", step=0.01)
    with col2:
        lon_input = st.number_input("Longitud", value=st.session_state.lon,
                                     format="%.5f", step=0.01)

    st.markdown("**3. API Key Mistral**")
    # Leer API key desde Streamlit Secrets (producción) o input manual (desarrollo)
    _key_secret = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    if _key_secret:
        api_key = _key_secret
        st.success("API Key configurada desde secrets.", icon="🔑")
    else:
        api_key = st.text_input("Mistral API Key", type="password",
                                 placeholder="Tu API key de Mistral AI",
                                 help="Obtené tu key gratis en console.mistral.ai")

    st.divider()
    generar = st.button("🌾 Generar informe", use_container_width=True)

    st.divider()
    st.markdown("**Opciones avanzadas**")
    anios = st.slider("Años de histórico ERA5", 5, 15, 10)
    usar_mistral_cultivos = st.checkbox("Estimar cultivos con Mistral AI",
                                         value=True,
                                         help="Si está desactivado usa cultivos por defecto según latitud")

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#888780;line-height:1.6'>
    🌐 Datos: ERA5 / Open-Meteo<br>
    🤖 Modelos: XGBoost · RF · SPI-3<br>
    ✍️ IA: Mistral Large<br>
    📍 GIS Dev Academy — gisdev.ar
    </div>
    """, unsafe_allow_html=True)

# ── Layout principal ───────────────────────────────────────────────────────
st.markdown("## AgroIA Report")
st.markdown("Ingresá las coordenadas del campo y generá el informe agroclimático completo con heladas, granizo y sequía.")

tab_mapa, tab_resultado, tab_pdf = st.tabs(["📍 Mapa y ubicación", "📊 Resultados", "📄 Informe PDF"])

# ── Tab 1: Mapa ────────────────────────────────────────────────────────────
with tab_mapa:
    st.markdown("**Hacé clic en el mapa para seleccionar el campo, o usá los inputs del panel lateral.**")

    m = folium.Map(location=[lat_input, lon_input], zoom_start=10,
                   tiles="CartoDB Positron")
    folium.Marker(
        location=[lat_input, lon_input],
        popup=f"{nombre or 'Campo'}<br>lat={lat_input:.4f}<br>lon={lon_input:.4f}",
        icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        location=[lat_input, lon_input],
        radius=2000, color="#1D9E75", fill=True, fill_opacity=0.1,
    ).add_to(m)

    resultado_mapa = st_folium(m, width="100%", height=450)

    # Capturar clic en el mapa
    if resultado_mapa and resultado_mapa.get("last_clicked"):
        click = resultado_mapa["last_clicked"]
        st.session_state.lat = round(click["lat"], 5)
        st.session_state.lon = round(click["lng"], 5)
        st.success(f"Campo actualizado: lat={st.session_state.lat}, lon={st.session_state.lon}")
        st.rerun()

    # Mostrar coordenadas actuales
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Latitud</div>
            <div class='metric-val'>{lat_input:.4f}°</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Longitud</div>
            <div class='metric-val'>{lon_input:.4f}°</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        zona = "Sur bonaerense" if lat_input < -36 else ("Centro" if lat_input < -32 else "Norte/NEA")
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Región estimada</div>
            <div class='metric-val' style='font-size:16px'>{zona}</div>
        </div>""", unsafe_allow_html=True)

# ── Proceso de generación ──────────────────────────────────────────────────
if generar:
    if not api_key:
        st.sidebar.error("Ingresá tu API Key de Mistral.")
    else:
        with st.spinner("Procesando... esto tarda 2-5 minutos"):
            progress = st.progress(0, text="Iniciando...")

            mensajes = []
            def cb(msg):
                mensajes.append(msg)
                pct = min(len(mensajes)*12, 90)
                progress.progress(pct, text=msg)

            try:
                # Pipeline ML
                cal, df_raw = pipeline_completo(
                    lat=lat_input, lon=lon_input,
                    nombre=nombre, anios=anios,
                    progress_cb=cb
                )
                progress.progress(55, text="Estimando cultivos...")

                # Cultivos
                if usar_mistral_cultivos and api_key:
                    cultivos = estimar_cultivos_mistral(lat_input, lon_input, nombre, api_key)
                else:
                    cultivos = cultivos_default(lat_input)

                progress.progress(65, text="Generando gráfico...")

                # Gráfico
                fig = generar_grafico(cal, cultivos, nombre or "Campo", lat_input, lon_input)

                progress.progress(75, text="Generando informe con Mistral AI...")

                # Informe
                resultado = generar_texto_mistral(
                    cal, cultivos, nombre or "Campo",
                    lat_input, lon_input, api_key, cb
                )
                texto = resultado[0]

                progress.progress(88, text="Generando PDF...")

                # PDF
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
                    fig_path = f_img.name
                fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="#F8F7F4")

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_pdf:
                    pdf_path = f_pdf.name

                generar_pdf(
                    nombre or "Campo", lat_input, lon_input,
                    *resultado,
                    fig_path=fig_path,
                    cultivos=cultivos,
                    output_path=pdf_path
                )

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                # Guardar en sesión
                st.session_state.cal      = cal
                st.session_state.cultivos = cultivos
                st.session_state.fig      = fig
                st.session_state.texto    = texto
                st.session_state.pdf_bytes= pdf_bytes

                progress.progress(100, text="Completado")
                time.sleep(0.5)
                progress.empty()
                st.success("Informe generado. Revisá las pestañas Resultados y PDF.")

            except Exception as e:
                progress.empty()
                st.error(f"Error: {e}")
                st.exception(e)

# ── Tab 2: Resultados ──────────────────────────────────────────────────────
with tab_resultado:
    if st.session_state.cal is None:
        st.info("Completá el formulario y hacé clic en Generar informe.")
    else:
        cal = st.session_state.cal
        fechas = __import__("pandas").date_range("2024-01-01", periods=365)
        MESES  = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

        dl = int((cal["clase"]>=1).sum()); da = int((cal["clase"]>=2).sum())
        ds_c = int((cal["cls_sequia"]>=1).sum()); dm_c = int((cal["cls_sequia"]>=2).sum())
        pg   = float(cal["p_gran"].max())
        spi  = float(cal["spi_clim"].min())

        st.markdown("### Resumen climático")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>Días helada leve/año</div>
                <div class='metric-val' style='color:#E24B4A'>{dl}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>Días helada alta/año</div>
                <div class='metric-val' style='color:#A32D2D'>{da}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>Prob. máx. granizo</div>
                <div class='metric-val' style='color:#185FA5'>{pg:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>Días sequía leve+</div>
                <div class='metric-val' style='color:#854F0B'>{ds_c}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Gráfico agroclimático anual")
        st.pyplot(st.session_state.fig, use_container_width=True)

        st.divider()
        st.markdown("### Calendario de riesgo por día")

        filtro = st.selectbox("Filtrar por amenaza",
                               ["Todos con riesgo","Solo heladas","Solo granizo","Solo sequía"])
        if filtro=="Solo heladas":
            df_show = cal[cal["clase"]>=1]
        elif filtro=="Solo granizo":
            df_show = cal[cal["p_gran"]>0.06]
        elif filtro=="Solo sequía":
            df_show = cal[cal["cls_sequia"]>=1]
        else:
            df_show = cal[(cal["clase"]>=1)|(cal["p_gran"]>0.06)|(cal["cls_sequia"]>=1)]

        df_display = df_show[["dia","tmin_clim","etiqueta","p_alto","p_gran","etq_sequia","spi_clim"]].copy()
        df_display.insert(0,"fecha",fechas[df_show["dia"].values-1].strftime("%d/%m"))
        df_display.columns=["Fecha","Tmin clim.","Helada","P(helada alta)","P(granizo)","Sequía","SPI-3"]
        df_display["P(helada alta)"] = df_display["P(helada alta)"].map("{:.1%}".format)
        df_display["P(granizo)"]     = df_display["P(granizo)"].map("{:.1%}".format)
        df_display["SPI-3"]          = df_display["SPI-3"].map("{:.2f}".format)
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=320)

        st.divider()
        st.markdown("### Cultivos estimados")
        import pandas as pd
        df_cult = pd.DataFrame([{
            "Cultivo":  c["cultivo"],
            "Tipo":     c["tipo"],
            "Siembra":  f"Día {c['sd_i']}–{c['sd_f']}",
            "Cosecha":  f"Día {c['cd_i']}–{c['cd_f']}",
            "S. helada": c.get("sh","-"),
            "S. granizo": c.get("sg","-"),
            "Fase crítica helada": c.get("fch","-"),
        } for c in st.session_state.cultivos])
        st.dataframe(df_cult, use_container_width=True)

        if st.session_state.texto:
            st.divider()
            with st.expander("Ver informe técnico completo", expanded=False):
                st.markdown(st.session_state.texto)

# ── Tab 3: PDF ─────────────────────────────────────────────────────────────
with tab_pdf:
    if st.session_state.pdf_bytes is None:
        st.info("Generá el informe primero desde el panel lateral.")
    else:
        nombre_archivo = f"AgroIA_{(nombre or 'Campo').replace(' ','_')}.pdf"
        st.success(f"PDF listo: **{nombre_archivo}**")

        col_a, col_b = st.columns([2,1])
        with col_a:
            st.download_button(
                label="⬇ Descargar PDF",
                data=st.session_state.pdf_bytes,
                file_name=nombre_archivo,
                mime="application/pdf",
                use_container_width=True,
            )
        with col_b:
            size_kb = len(st.session_state.pdf_bytes)//1024
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>Tamaño</div>
                <div class='metric-val' style='font-size:18px'>{size_kb} KB</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Contenido del PDF")
        st.markdown("""
        El informe incluye:
        - **Portada** con nombre del campo y coordenadas
        - **Tabla resumen** con las 3 amenazas (helada, granizo, sequía)
        - **Períodos de riesgo** con fechas exactas
        - **Tabla de cultivos** con sensibilidades y fases críticas
        - **Gráfico de 4 paneles** (Gantt + heladas + granizo + sequía)
        - **Informe técnico agronómico** generado por Mistral AI (500-700 palabras)
        - **Pie de página** con fuentes de datos y modelos utilizados
        """)

        st.divider()
        st.markdown("### Compartir / Enviar")
        st.markdown("Podés adjuntar este PDF directamente a un email o WhatsApp para tu cliente.")
        email_dest = st.text_input("Email del destinatario (opcional)", placeholder="agrónomo@ejemplo.com")
        if email_dest:
            st.info(f"Para envío automático por email integrá SendGrid o Amazon SES a esta app.")
