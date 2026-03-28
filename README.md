# AgroIA Report — GIS Dev Academy

App Streamlit para generar calendarios agroclimáticos anuales con IA.

## Estructura

```
agroai_streamlit/
├── app.py              ← App principal Streamlit
├── agro_engine.py      ← Motor de datos y modelos ML
├── agro_charts.py      ← Gráficos y estimación de cultivos
├── agro_report.py      ← Generación de PDF e informe Mistral
├── requirements.txt    ← Dependencias
└── README.md
```

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Cloud (gratis)

1. Subí este directorio a un repo GitHub
2. Entrá a share.streamlit.io
3. Conectá tu repo y apuntá a app.py
4. Listo — URL pública en 2 minutos

## Variables de entorno (opcional)

Para no ingresar la API key en cada sesión, podés agregarla en
Streamlit Cloud → Settings → Secrets:

```toml
MISTRAL_API_KEY = "tu_key_aqui"
```

Y en app.py leerla con:
```python
import streamlit as st
api_key = st.secrets.get("MISTRAL_API_KEY", "")
```

## Cómo funciona

1. El usuario selecciona el campo en el mapa Leaflet (streamlit-folium)
2. `agro_engine.py` descarga 10 años ERA5 y entrena XGBoost + SPI-3
3. `agro_charts.py` genera el gráfico de 4 paneles con matplotlib
4. `agro_report.py` llama a Mistral AI y construye el PDF con reportlab
5. El usuario descarga el PDF desde la pestaña "Informe PDF"

## GIS Dev Academy — gisdev.ar
