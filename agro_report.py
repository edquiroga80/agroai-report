# agro_report.py — Generación de PDF e informe con Mistral
# GIS Dev Academy — AgroIA

import re
import os
import importlib, inspect, pkgutil
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

MESES = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


def _make_mistral_client(api_key):
    import mistralai
    candidatos = []
    for _, modname, _ in pkgutil.walk_packages(
            path=mistralai.__path__, prefix="mistralai.", onerror=lambda x: None):
        try:
            mod = importlib.import_module(modname)
            for n, obj in inspect.getmembers(mod, inspect.isclass):
                if "client" in n.lower() or n == "Mistral":
                    candidatos.append((n, obj))
        except Exception:
            pass
    for n, c in candidatos:
        if n == "Mistral": return c(api_key=api_key)
    for n, c in candidatos:
        if "mistral" in n.lower(): return c(api_key=api_key)
    raise ImportError("No se encontró clase cliente en mistralai.")


def _chat(client, prompt, temperature=0.25, max_tokens=2200):
    msgs = [{"role":"user","content":prompt}]
    try:
        r = client.chat.complete(model="mistral-large-latest",
                                  messages=msgs, temperature=temperature,
                                  max_tokens=max_tokens)
    except AttributeError:
        r = client.chat(model="mistral-large-latest",
                        messages=msgs, temperature=temperature,
                        max_tokens=max_tokens)
    return r.choices[0].message.content


def _dias_riesgo(cal):
    fechas = pd.date_range("2024-01-01", periods=365)
    lineas_h, lineas_g, lineas_s = [], [], []

    en = False; ini = None; cr = 0
    for _, row in cal.iterrows():
        d = int(row["dia"]); cls = int(row["clase"])
        if cls>=1 and not en:
            en=True; ini=d; cr=cls
        elif cls>=1 and en:
            cr=max(cr,cls)
        elif cls==0 and en:
            fin=d-1
            tag="ALTO/HELADA" if cr>=2 else "leve"
            lineas_h.append(f"  {fechas[ini-1].strftime('%d/%m')} - "
                            f"{fechas[fin-1].strftime('%d/%m')}: riesgo {tag} ({fin-ini+1} días)")
            en=False; cr=0
    if en:
        tag="ALTO/HELADA" if cr>=2 else "leve"
        lineas_h.append(f"  {fechas[ini-1].strftime('%d/%m')} - "
                        f"{fechas[364].strftime('%d/%m')}: riesgo {tag}")

    for mes in range(1,13):
        sub=cal[cal["mes"]==mes]
        pm=float(sub["p_gran"].max()) if len(sub) else 0
        if pm>0.05:
            lineas_g.append(f"  {MESES[mes]}: prob. máxima {pm:.1%}")

    for mes in range(1,13):
        sub=cal[cal["mes"]==mes]
        psm=float(sub["p_seq_mod"].mean()) if len(sub) else 0
        pss=float(sub["p_seq_sev"].mean()) if len(sub) else 0
        spi=float(sub["spi_clim"].mean()) if len(sub) else 0
        bal=float(sub["bal_clim"].mean()) if len(sub) else 0
        if psm+pss>0.12 or spi<-0.5:
            niv="severa" if pss>0.15 else "moderada" if psm>0.15 else "leve"
            lineas_s.append(f"  {MESES[mes]}: sequía {niv} — SPI-3={spi:.2f}, balance={bal:.0f}mm")

    return (
        "\n".join(lineas_h) or "  Sin períodos de riesgo detectados.",
        "\n".join(lineas_g) or "  Sin meses con riesgo significativo.",
        "\n".join(lineas_s) or "  Sin meses con déficit hídrico significativo.",
    )


def generar_texto_mistral(cal, cultivos, nombre, lat, lon, api_key, progress_cb=None):
    """Genera el informe técnico con Mistral AI."""
    if progress_cb: progress_cb("Generando informe con Mistral AI...")

    dl=int((cal["clase"]>=1).sum()); da=int((cal["clase"]>=2).sum())
    ds=int((cal["cls_sequia"]>=1).sum()); dm=int((cal["cls_sequia"]>=2).sum())
    tmin=float(cal["tmin_clim"].min())
    spi_min=float(cal["spi_clim"].min())
    fechas=pd.date_range("2024-01-01",periods=365)
    p0=fechas[int(cal[cal["clase"]>=1].iloc[0]["dia"])-1].strftime("%d/%m") if dl>0 else "-"
    p1=fechas[int(cal[cal["clase"]>=1].iloc[-1]["dia"])-1].strftime("%d/%m") if dl>0 else "-"
    a0=fechas[int(cal[cal["clase"]>=2].iloc[0]["dia"])-1].strftime("%d/%m") if da>0 else "-"
    a1=fechas[int(cal[cal["clase"]>=2].iloc[-1]["dia"])-1].strftime("%d/%m") if da>0 else "-"
    pg_max=float(cal["p_gran"].max())
    mes_g=int(cal.loc[cal["p_gran"].idxmax(),"mes"])
    mes_s=int(cal.loc[cal["spi_clim"].idxmin(),"mes"])

    dh,dg,ds_txt = _dias_riesgo(cal)
    cult_str="\n".join([
        f"  - {c['cultivo']}: helada={c['sh']}, granizo={c['sg']}. "
        f"Crítico helada: {c['fch']}. Crítico granizo: {c['fcg']}."
        for c in cultivos if c["tipo"]!="perenne"
    ])

    prompt=(
        f"Sos ingeniero agrónomo especialista en climatología aplicada al agro argentino.\n\n"
        f"CAMPO: {nombre} | lat={lat:.4f}, lon={lon:.4f}\n\n"
        f"RESUMEN CLIMÁTICO (modelo ML + 10 años ERA5):\n"
        f"  Heladas riesgo leve (3-5°C): {dl} días/año, ventana: {p0} a {p1}\n"
        f"  Heladas riesgo alto (≤3°C): {da} días/año, ventana: {a0} a {a1}\n"
        f"  Tmin climatológica mínima: {tmin:.1f}°C\n"
        f"  Granizo: prob. máx. mensual {pg_max:.1%} en {MESES[mes_g]}\n"
        f"  Sequía leve+: {ds} días/año | Moderada+: {dm} días\n"
        f"  SPI-3 mínimo: {spi_min:.2f} en {MESES[mes_s]}\n\n"
        f"PERÍODOS DE RIESGO DE HELADAS:\n{dh}\n\n"
        f"MESES CON RIESGO DE GRANIZO:\n{dg}\n\n"
        f"MESES CON DÉFICIT HÍDRICO:\n{ds_txt}\n\n"
        f"CULTIVOS:\n{cult_str}\n\n"
        f"Redactá un INFORME TÉCNICO AGRONÓMICO de 800-1000 palabras en español. "
        f"Es OBLIGATORIO completar TODAS las secciones hasta el final sin cortar. "
        f"con secciones en mayúsculas:\n"
        f"1. CARACTERIZACIÓN AGROCLIMATICA\n"
        f"2. RIESGO DE HELADAS POR CULTIVO\n"
        f"3. RIESGO DE GRANIZO\n"
        f"4. RIESGO DE SEQUÍA Y DÉFICIT HÍDRICO\n"
        f"5. RECOMENDACIONES TÉCNICAS INTEGRADAS\n"
        f"6. OPORTUNIDADES PRODUCTIVAS\n"
        f"Usá terminología técnica precisa con datos cuantitativos. "
        f"Dirigido a productores y asesores rurales profesionales."
    )

    try:
        client = _make_mistral_client(api_key)
        texto = _chat(client, prompt)
    except Exception as e:
        texto=(f"[Error Mistral: {e}]\n\nResumen automático:\n"
               f"Heladas: {dl} días leve, {da} días alto. Ventana: {p0} a {p1}.\n"
               f"Granizo: {pg_max:.1%} en {MESES[mes_g]}. "
               f"Sequía SPI-3 mín={spi_min:.2f} en {MESES[mes_s]}.")

    return texto, dh, dg, ds_txt, dl, da, p0, p1, a0, a1, pg_max, MESES[mes_g], ds, dm, spi_min, MESES[mes_s]


def generar_pdf(nombre, lat, lon, texto, dh, dg, ds_txt,
                dl, da, p0, p1, a0, a1, pg_max, mes_g,
                ds, dm, spi_min, mes_s,
                fig_path, cultivos, output_path):
    """Genera el PDF completo y lo guarda en output_path."""

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    S = {
        "t":  ParagraphStyle("t",  fontName="Helvetica-Bold", fontSize=16,
                              textColor=colors.HexColor("#1D9E75"),
                              spaceAfter=4, alignment=TA_CENTER),
        "s":  ParagraphStyle("s",  fontName="Helvetica", fontSize=10,
                              textColor=colors.HexColor("#888780"),
                              spaceAfter=12, alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
                              textColor=colors.HexColor("#333330"),
                              spaceBefore=10, spaceAfter=4),
        "b":  ParagraphStyle("b",  fontName="Helvetica", fontSize=9.5,
                              leading=14, textColor=colors.HexColor("#333330"),
                              spaceAfter=6, alignment=TA_JUSTIFY),
        "m":  ParagraphStyle("m",  fontName="Courier", fontSize=8.5,
                              leading=13, textColor=colors.HexColor("#444441"),
                              spaceAfter=3),
        "p":  ParagraphStyle("p",  fontName="Helvetica", fontSize=7.5,
                              textColor=colors.HexColor("#888780"),
                              alignment=TA_CENTER),
    }

    def tabla_estilo(data, col_widths, header_color="#1D9E75"):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), colors.HexColor(header_color)),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F8F7F4"), colors.HexColor("#E1F5EE")]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#D3D1C7")),
            ("LEFTPADDING",   (0,0),(-1,-1), 7),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        return t

    story = []
    story.append(Paragraph("Informe Técnico Agronómico", S["t"]))
    story.append(Paragraph(
        f"{nombre}  |  lat={lat:.4f}, lon={lon:.4f}  |  GIS Dev Academy — AgroIA",
        S["s"]))
    story.append(HRFlowable(width="100%",thickness=1,
                             color=colors.HexColor("#1D9E75"),spaceAfter=10))

    # Tabla resumen
    story.append(Paragraph("Resumen Climático Integrado", S["h2"]))
    tdata=[
        ["Amenaza","Indicador","Valor"],
        ["Helada","Días riesgo leve (3-5°C)",f"{dl} días/año"],
        ["","Ventana riesgo leve",f"{p0}  →  {p1}"],
        ["","Días riesgo alto (≤3°C)",f"{da} días/año"],
        ["","Ventana riesgo alto",f"{a0}  →  {a1}"],
        ["Granizo","Prob. máxima mensual",f"{pg_max:.1%}  ({mes_g})"],
        ["Sequía","Días sequía leve+/año",f"{ds} días"],
        ["","Días sequía moderada+/año",f"{dm} días"],
        [f"",f"SPI-3 mínimo ({mes_s})",f"{spi_min:.2f}"],
    ]
    t=Table(tdata,colWidths=[3.5*cm,8*cm,4.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#333330")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("SPAN",(0,1),(0,4)),("SPAN",(0,6),(0,8)),
        ("BACKGROUND",(0,1),(0,4),colors.HexColor("#E24B4A")),
        ("TEXTCOLOR",(0,1),(0,4),colors.white),
        ("FONTNAME",(0,1),(0,4),"Helvetica-Bold"),
        ("BACKGROUND",(0,5),(0,5),colors.HexColor("#185FA5")),
        ("TEXTCOLOR",(0,5),(0,5),colors.white),
        ("FONTNAME",(0,5),(0,5),"Helvetica-Bold"),
        ("BACKGROUND",(0,6),(0,8),colors.HexColor("#854F0B")),
        ("TEXTCOLOR",(0,6),(0,8),colors.white),
        ("FONTNAME",(0,6),(0,8),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#F8F7F4"),colors.HexColor("#E1F5EE")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D3D1C7")),
        ("LEFTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1,0.4*cm))

    for titulo, lines in [
        ("Períodos de Riesgo de Heladas", dh.split("\n")),
        ("Meses con Riesgo de Granizo",   dg.split("\n")),
        ("Meses con Déficit Hídrico / Sequía (SPI-3)", ds_txt.split("\n")),
    ]:
        story.append(Paragraph(titulo, S["h2"]))
        for l in lines:
            story.append(Paragraph(l.strip(), S["m"]))
        story.append(Spacer(1,0.25*cm))

    # Tabla cultivos
    story.append(Paragraph("Cultivos de la Zona y Sensibilidad", S["h2"]))
    cdata=[["Cultivo","Tipo","Helada","Granizo","Fase crítica helada"]]
    for c in cultivos:
        cdata.append([c["cultivo"],c["tipo"],c.get("sh","-"),
                      c.get("sg","-"),c.get("fch","-")])
    story.append(tabla_estilo(cdata,[3.5*cm,2.5*cm,2.5*cm,2.5*cm,5.5*cm],"#333330"))
    story.append(Spacer(1,0.5*cm))

    # Gráfico
    if fig_path and os.path.exists(fig_path):
        story.append(Paragraph("Gráfico Agroclimático Anual", S["h2"]))
        story.append(Image(fig_path, width=17*cm, height=11.5*cm))
        story.append(Spacer(1,0.4*cm))

    # Informe Mistral
    story.append(HRFlowable(width="100%",thickness=0.5,
                             color=colors.HexColor("#D3D1C7"),spaceAfter=8))
    story.append(Paragraph("Informe Técnico Agronómico Integrado", S["h2"]))

    def md(txt):
        txt=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',txt)
        txt=re.sub(r'__(.+?)__',r'<b>\1</b>',txt)
        txt=re.sub(r'\*(.+?)\*',r'<i>\1</i>',txt)
        return txt

    def es_tab(l):  return l.startswith("|") and l.endswith("|") and l.count("|")>=2
    def es_sep(l):  return bool(re.match(r'^[|:\-\s]+$',l)) and "-" in l and "|" in l

    def render_tabla(rows):
        filas=[]
        for r in rows:
            if es_sep(r.strip()): continue
            filas.append([md(c.strip()) for c in r.strip().strip("|").split("|")])
        if not filas: return None
        nc=max(len(f) for f in filas)
        data=[]
        for f in filas:
            while len(f)<nc: f.append("")
            data.append(f[:nc])
        cws=[3*cm]+[(14*cm)/(nc-1)]*(nc-1) if nc>1 else [17*cm]
        return tabla_estilo(data,cws)

    lineas=texto.split("\n"); bloques=[]; i=0
    while i<len(lineas):
        l=lineas[i].strip()
        if es_tab(l):
            rows=[]
            while i<len(lineas):
                li=lineas[i].strip()
                if es_tab(li) or es_sep(li): rows.append(li); i+=1
                elif li=="": break
                else: break
            bloques.append(("tabla",rows))
        else:
            bloques.append(("linea",l)); i+=1

    par=[]
    def flush():
        if par:
            story.append(Paragraph(md(" ".join(par)),S["b"]))
            par.clear()

    for tipo,cont in bloques:
        if tipo=="tabla":
            flush()
            t2=render_tabla(cont)
            if t2:
                story.append(Spacer(1,0.2*cm))
                story.append(t2)
                story.append(Spacer(1,0.2*cm))
            continue
        l=cont
        if re.match(r'^-{3,}$',l) or l in ("***","___"): flush(); continue
        m2=re.match(r'^#{1,4}\s*(.*)',l)
        if m2:
            flush()
            story.append(Paragraph(re.sub(r'\*+','',m2.group(1)).strip(),S["h2"]))
            continue
        if re.match(r'^\d+\.\s+[A-ZÁÉÍÓÚ]',l):
            flush()
            story.append(Paragraph(re.sub(r'\*+','',re.sub(r'^\d+\.\s+','',l)).strip(),S["h2"]))
            continue
        if not l: flush(); story.append(Spacer(1,0.1*cm)); continue
        if l.startswith("- "): flush(); story.append(Paragraph(f"• {md(l[2:])}",S["b"])); continue
        par.append(l)
    flush()

    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,
                             color=colors.HexColor("#D3D1C7"),spaceAfter=4))
    story.append(Paragraph(
        "GIS Dev Academy — AgroIA  |  Datos: ERA5 / Open-Meteo  |  "
        "Modelos: XGBoost + RF + SPI-3  |  IA: Mistral",
        S["p"]))

    doc.build(story)
    return output_path
