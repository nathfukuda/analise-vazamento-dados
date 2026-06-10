# ============================================================
# NÍVEL 4 — Relatório PDF com Storytelling (versão refinada)
# Capa em canvas, rodapé com paginação, índice, seções com
# barra de destaque e gráficos repaginados. Pronto p/ portfólio.
# ============================================================

import sys, os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, HRFlowable,
                                KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Paleta ----
RED    = "#c0392b"
NAVY   = "#2c3e50"
SLATE  = "#5d6d7e"
LIGHT  = "#f4f6f7"
CREAM  = "#fdf2e9"
GREY   = "#95a5a6"

IMGDIR = "outputs/report_images"
os.makedirs(IMGDIR, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.alpha": .22, "axes.axisbelow": True,
    "axes.edgecolor": "#999", "axes.linewidth": .8, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": NAVY,
})

df      = pd.read_csv("data/processed/breaches_ml.csv")
ml      = json.load(open("data/processed/n3_ml_resultados.json", encoding="utf-8"))
season  = pd.read_csv("data/processed/n1_sazonalidade.csv")
by_method = pd.read_csv("data/processed/n1_por_metodo.csv")
reincid = pd.read_csv("data/processed/n1_reincidentes.csv")

# ---- KPIs ----
total_inc = len(df)
total_rec = df["records"].sum()
periodo   = f"{df.year.min()}-{df.year.max()}"
n_reincid = len(reincid)

# ============================================================
# Helpers de gráfico
# ============================================================
def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def save(fig, name):
    path = os.path.join(IMGDIR, name)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def label_barh(ax, fmt="{:.0f}", pad=0.01):
    xmax = max((b.get_width() for b in ax.patches), default=0)
    for b in ax.patches:
        ax.text(b.get_width() + xmax*pad, b.get_y() + b.get_height()/2,
                fmt.format(b.get_width()), va="center", ha="left",
                fontsize=8.5, color=NAVY)

# Fig 1 — evolução anual (eixo duplo)
g = df.groupby("year").agg(inc=("entity", "size"), rec=("records", "sum")).reset_index()
fig, ax = plt.subplots(figsize=(8, 3.3))
ax.bar(g.year, g.inc, color=RED, alpha=.85, label="Incidentes", zorder=3)
ax.set_ylabel("Incidentes", color=RED); ax.set_xlabel("Ano")
ax.tick_params(axis="y", colors=RED); despine(ax)
ax2 = ax.twinx()
ax2.plot(g.year, g.rec/1e9, color=NAVY, marker="o", ms=4, lw=2.2,
         label="Registros (bi)")
ax2.set_ylabel("Registros expostos (bilhões)", color=NAVY)
ax2.tick_params(axis="y", colors=NAVY)
ax2.spines["top"].set_visible(False)
ax.set_title("Evolução dos vazamentos por ano")
f_ano = save(fig, "ano.png")

# Fig 2 — sazonalidade
order = season.sort_values("month")
fig, ax = plt.subplots(figsize=(8, 3.0))
bars = ax.bar(order.mes, order.incidentes,
              color=plt.cm.YlOrRd(np.linspace(.4, .9, len(order))), zorder=3)
ax.bar_label(bars, fontsize=8.5, color=NAVY, padding=2)
ax.set_title("Sazonalidade — incidentes por mes do ano")
ax.set_ylabel("Incidentes"); ax.margins(y=.15); despine(ax)
f_season = save(fig, "season.png")

# Fig 3 — registros por método
m = by_method.sort_values("total_registros")
fig, ax = plt.subplots(figsize=(8, 3.0))
ax.barh(m.method, m.total_registros/1e9,
        color=plt.cm.Reds(np.linspace(.45, .9, len(m))), zorder=3)
label_barh(ax, "{:.1f} bi"); ax.margins(x=.12)
ax.set_title("Volume de registros expostos por metodo (bilhoes)")
ax.set_xlabel("Registros (bilhoes)"); despine(ax)
f_method = save(fig, "method.png")

# Fig 4 — entidades reincidentes
r = reincid.sort_values("ataques").tail(12)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(r.entity, r.ataques,
        color=plt.cm.Purples(np.linspace(.45, .9, len(r))), zorder=3)
label_barh(ax, "{:.0f}"); ax.margins(x=.1)
ax.set_title("Entidades com multiplos ataques (Top 12)")
ax.set_xlabel("No de ataques")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True)); despine(ax)
f_reincid = save(fig, "reincid.png")

# Fig 5 — clusters (PCA)
CLU_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
fig, ax = plt.subplots(figsize=(7.6, 3.7))
for i, c in enumerate(sorted(df.cluster.unique())):
    sub = df[df.cluster == c]
    ax.scatter(sub.pc1, sub.pc2, s=20, alpha=.7,
               color=CLU_COLORS[i % len(CLU_COLORS)], label=f"Cluster {c}",
               edgecolor="white", linewidth=.3, zorder=3)
ax.set_title("Perfis de incidentes (K-Means + PCA)")
ax.set_xlabel("Componente principal 1"); ax.set_ylabel("Componente principal 2")
ax.legend(frameon=False, fontsize=9); despine(ax)
f_cluster = save(fig, "cluster.png")

# Fig 6 — anomalias
fig, ax = plt.subplots(figsize=(7.6, 3.7))
norm = df[df.anomaly == 1]; an = df[df.anomaly == -1]
ax.scatter(norm.year, np.log10(norm.records.clip(lower=1)), s=16, c="#3498db",
           alpha=.45, label="Normal", zorder=2)
ax.scatter(an.year, np.log10(an.records.clip(lower=1)), s=55, c=RED,
           edgecolor="k", lw=.6, label="Anomalia", zorder=3)
ax.set_title("Deteccao de anomalias (Isolation Forest)")
ax.set_xlabel("Ano"); ax.set_ylabel("log10(registros)")
ax.legend(frameon=False, fontsize=9); despine(ax)
f_anom = save(fig, "anom.png")
print("OK - imagens do relatorio geradas")

# ============================================================
# Insights calculados
# ============================================================
mes_top      = season.loc[season.incidentes.idxmax(), "mes"]
mes_top_n    = int(season.incidentes.max())
metodo_top   = by_method.iloc[0]
hacked_share = (df.method == "Hacked").mean()
entidade_top = reincid.iloc[0]
acc  = ml["classificacao"]["acuracia"]
base = ml["classificacao"]["baseline"]

# ============================================================
# Decorações de página (canvas)
# ============================================================
W, H = A4

def _cover(canvas, doc):
    canvas.saveState()
    # banda superior navy
    canvas.setFillColor(colors.HexColor(NAVY))
    canvas.rect(0, H-9.5*cm, W, 9.5*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(RED))
    canvas.rect(0, H-9.7*cm, W, 0.22*cm, fill=1, stroke=0)
    # rótulo
    canvas.setFillColor(colors.HexColor("#e67e22"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(W/2, H-2.7*cm, "RELATORIO ANALITICO DE SEGURANCA")
    # título
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 30)
    canvas.drawCentredString(W/2, H-4.3*cm, "Maiores Vazamentos")
    canvas.drawCentredString(W/2, H-5.6*cm, "de Dados do Mundo")
    # subtítulo
    canvas.setFillColor(colors.HexColor("#bdc3c7"))
    canvas.setFont("Helvetica", 12)
    canvas.drawCentredString(W/2, H-6.8*cm,
        "Analise Exploratoria, Machine Learning e Insights de Seguranca")
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(W/2, H-8.0*cm,
        f"Periodo {periodo}   |   {total_inc} incidentes   |   "
        f"{total_rec/1e9:.1f} bilhoes de registros expostos")

    # KPI cards
    cards = [(f"{total_inc}", "Incidentes"),
             (f"{total_rec/1e9:.1f} bi", "Registros expostos"),
             (f"{n_reincid}", "Reincidentes"),
             (periodo, "Periodo")]
    cw, ch, gap = 3.9*cm, 2.3*cm, 0.4*cm
    total_w = len(cards)*cw + (len(cards)-1)*gap
    x0 = (W - total_w)/2
    y0 = H-13.2*cm
    for i, (val, lab) in enumerate(cards):
        x = x0 + i*(cw+gap)
        canvas.setFillColor(colors.HexColor(LIGHT))
        canvas.roundRect(x, y0, cw, ch, 8, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(RED))
        canvas.setFont("Helvetica-Bold", 19)
        canvas.drawCentredString(x+cw/2, y0+ch-1.15*cm, val)
        canvas.setFillColor(colors.HexColor(SLATE))
        canvas.setFont("Helvetica", 8.5)
        canvas.drawCentredString(x+cw/2, y0+0.45*cm, lab.upper())

    # rodapé da capa
    canvas.setStrokeColor(colors.HexColor("#dfe3e6"))
    canvas.setLineWidth(.8)
    canvas.line(3*cm, 2.2*cm, W-3*cm, 2.2*cm)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.setFont("Helvetica", 8.5)
    canvas.drawCentredString(W/2, 1.6*cm,
        "Gerado com Python  -  Pandas  -  scikit-learn  -  Plotly  -  ReportLab")
    canvas.restoreState()

def _page(canvas, doc):
    canvas.saveState()
    # barra de topo fina
    canvas.setFillColor(colors.HexColor(RED))
    canvas.rect(0, H-0.3*cm, W, 0.3*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, H-0.95*cm, "Maiores Vazamentos de Dados do Mundo")
    # rodapé
    canvas.setStrokeColor(colors.HexColor("#e3e6e8"))
    canvas.setLineWidth(.6)
    canvas.line(2*cm, 1.35*cm, W-2*cm, 1.35*cm)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 0.95*cm, "Relatorio Analitico de Seguranca")
    canvas.drawRightString(W-2*cm, 0.95*cm, f"Pagina {doc.page-1}")
    canvas.restoreState()

# ============================================================
# Estilos
# ============================================================
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H2", fontSize=15, textColor=colors.HexColor(NAVY),
           spaceBefore=2, spaceAfter=2, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15.5,
           alignment=TA_JUSTIFY, spaceAfter=7, textColor=colors.HexColor("#2b2b2b")))
styles.add(ParagraphStyle("Lead", parent=styles["Normal"], fontSize=11.5, leading=17,
           alignment=TA_JUSTIFY, spaceAfter=9, textColor=colors.HexColor(NAVY)))
styles.add(ParagraphStyle("Insight", parent=styles["Normal"], fontSize=10.3, leading=15,
           leftIndent=8, textColor=colors.HexColor(NAVY),
           backColor=colors.HexColor(CREAM), borderColor=colors.HexColor("#f0c9a8"),
           borderWidth=.6, borderPadding=8, spaceAfter=9, spaceBefore=2))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5,
           textColor=colors.HexColor(GREY), alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("TocItem", parent=styles["Normal"], fontSize=11, leading=20,
           textColor=colors.HexColor(NAVY)))

S = []
def p(txt, st="Body"): S.append(Paragraph(txt, styles[st]))
def gap(x=6): S.append(Spacer(1, x))
def cap(txt): S.append(Paragraph(txt, styles["Cap"]))

def h(num, txt):
    block = [
        Paragraph(f'<font color="{RED}" size="15"><b>{num}</b></font>'
                  f'&nbsp;&nbsp;<b>{txt}</b>', styles["H2"]),
        HRFlowable(width="100%", color=colors.HexColor(RED), thickness=1.3,
                   spaceBefore=3, spaceAfter=9),
    ]
    S.append(KeepTogether(block))

def fig_img(path, w=16, ratio=0.40):
    S.append(Image(path, width=w*cm, height=w*cm*ratio))

# ============================================================
# CAPA (desenhada em _cover) — página em branco que recebe o canvas
# ============================================================
S.append(Spacer(1, 1)); S.append(PageBreak())

# ============================================================
# ÍNDICE
# ============================================================
gap(6)
p("Neste relatorio", "H2")
S.append(HRFlowable(width="100%", color=colors.HexColor(RED), thickness=1.3,
                    spaceBefore=3, spaceAfter=12))
toc = [("1", "Sumario executivo"),
       ("2", "Quando os ataques acontecem (sazonalidade)"),
       ("3", "Como os dados vazam (metodos)"),
       ("4", "Quem e alvo recorrente (reincidencia)"),
       ("5", "O que os modelos aprenderam (machine learning)"),
       ("6", "Insights e recomendacoes")]
for n, t in toc:
    S.append(Paragraph(
        f'<font color="{RED}"><b>{n}</b></font>&nbsp;&nbsp;&nbsp;{t}', styles["TocItem"]))
gap(16)
p("<b>Sobre o conjunto de dados.</b> A base reune os maiores vazamentos de dados "
  f"publicamente documentados entre {periodo}, totalizando {total_inc} incidentes. "
  "Cada registro descreve a organizacao afetada, o setor, o metodo de ataque, o volume "
  "de dados expostos, a sensibilidade da informacao e a data do incidente.", "Body")
S.append(PageBreak())

# ============================================================
# 1. SUMÁRIO EXECUTIVO
# ============================================================
h("1", "Sumario executivo")
p(f"Entre {periodo}, foram catalogados <b>{total_inc} grandes vazamentos de dados</b>, "
  f"expondo o equivalente a <b>{total_rec/1e9:.1f} bilhoes de registros</b>. A analise revela "
  f"uma tendencia clara: os incidentes nao so se tornaram mais frequentes ao longo da decada, "
  f"como cresceram em <b>magnitude</b> -- megavazamentos de centenas de milhoes de registros, "
  f"raros no inicio do periodo, tornaram-se recorrentes a partir de 2018.", "Lead")
p(f"O ataque por invasao direta (<b>Hacked</b>) domina o cenario, respondendo por "
  f"<b>{hacked_share:.0%}</b> de todos os incidentes e pelo maior volume de dados expostos "
  f"(<b>{metodo_top['total_bi']} bilhoes</b> de registros). Falhas humanas e de configuracao, "
  f"porem, permanecem relevantes e frequentemente evitaveis -- um lembrete de que parte "
  f"importante do risco nao vem de adversarios sofisticados, mas de controles basicos ausentes.")
gap(2)
fig_img(f_ano, ratio=0.42)
cap("Figura 1 - Frequencia (barras) e volume exposto (linha) por ano. "
    "A divergencia entre as duas curvas evidencia o aumento da magnitude por incidente.")

# ============================================================
# 2. SAZONALIDADE
# ============================================================
S.append(PageBreak())
h("2", "Quando os ataques acontecem")
p(f"Ao decompor os incidentes por mes, observa-se concentracao em <b>{mes_top}</b> "
  f"({mes_top_n} incidentes), seguido pelos meses de meio e fim de ano. Embora vazamentos "
  f"ocorram o ano inteiro, esse padrao sugere janelas de maior exposicao -- uteis para "
  f"planejar campanhas de conscientizacao e reforco de monitoramento.")
fig_img(f_season, ratio=0.38)
cap("Figura 2 - Distribuicao dos incidentes ao longo dos meses do ano.")
p("<b>Leitura de negocio:</b> a sazonalidade nao substitui vigilancia continua, mas indica "
  "periodos em que as equipes de seguranca devem elevar o nivel de alerta e antecipar "
  "auditorias e simulacoes de incidente.", "Insight")

# ============================================================
# 3. MÉTODOS DE ATAQUE
# ============================================================
S.append(PageBreak())
h("3", "Como os dados vazam")
p(f"Nem todo metodo e igualmente destrutivo. O <b>{metodo_top['method']}</b> lidera tanto em "
  f"frequencia quanto em volume, mas categorias como <b>Poor Security</b> (ma configuracao) e "
  f"<b>Lost / Stolen Device</b> mostram que parte expressiva das exposicoes decorre de falhas "
  f"operacionais -- nao de ataques sofisticados.")
fig_img(f_method, ratio=0.38)
cap("Figura 3 - Total de registros expostos agregado por metodo de ataque.")
p("<b>Insight:</b> uma fracao relevante dos vazamentos e evitavel com controles basicos -- "
  "criptografia de dispositivos, gestao de acessos e revisao de configuracoes de nuvem. "
  "Esses controles tem alto retorno por exigirem investimento comparativamente baixo.", "Insight")

# ============================================================
# 4. REINCIDÊNCIA
# ============================================================
S.append(PageBreak())
h("4", "Quem e alvo recorrente")
p(f"De todas as organizacoes analisadas, <b>{n_reincid}</b> sofreram mais de um vazamento no "
  f"periodo. A lider, <b>{entidade_top['entity']}</b>, acumula <b>{int(entidade_top['ataques'])} "
  f"incidentes</b>. A reincidencia concentra-se em grandes plataformas digitais, refletindo tanto "
  f"a superficie de ataque ampliada quanto o valor dos dados que custodiam.")
fig_img(f_reincid, ratio=0.44)
cap("Figura 4 - Organizacoes com mais de um vazamento documentado no periodo.")
p("<b>Implicacao:</b> vazar mais de uma vez sugere causa-raiz nao resolvida. Essas "
  "organizacoes merecem um processo de pos-incidente estruturado e acompanhamento "
  "continuo, em vez de remediacao pontual.", "Insight")

# ============================================================
# 5. MACHINE LEARNING
# ============================================================
S.append(PageBreak())
h("5", "O que os modelos aprenderam")
p("<b>5.1  Classificacao -- e possivel prever o metodo de ataque?</b>", "Body")
p(f"Treinamos um Random Forest para prever o metodo a partir de setor, ano e tamanho do "
  f"vazamento. O modelo atingiu <b>{acc:.0%}</b> de acuracia, <i>abaixo</i> do baseline ingenuo "
  f"de {base:.0%} (sempre prever 'Hacked'). Esse resultado e, em si, o achado: <b>setor, ano e "
  f"volume nao determinam o metodo de ataque</b> -- a escolha do vetor depende de fatores nao "
  f"capturados nesses atributos (postura de seguranca, motivacao do atacante, oportunidade). "
  f"Honestidade analitica vale mais que uma metrica inflada.")
gap(2)
p("<b>5.2  Clustering -- perfis de incidente</b>", "Body")
perf = ml["clustering"]["perfis"]
clu_rows = [["Cluster", "Incid.", "Ano med.", "Reg. medianos", "Sensib.", "Setor top"]]
for rr in perf:
    clu_rows.append([str(rr["cluster"]), str(rr["incidentes"]), str(rr["ano_medio"]),
                     f"{int(rr['registros_medianos']):,}".replace(",", "."),
                     str(rr["sensibilidade_media"]), str(rr["setor_top"])])
t = Table(clu_rows, colWidths=[1.9*cm, 1.7*cm, 2.1*cm, 3.4*cm, 1.9*cm, 2.7*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.7), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(RED)),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
    ("LINEBELOW", (0, 1), (-1, -2), .4, colors.HexColor("#dfe3e6"))]))
S.append(t); gap(6)
p(f"O K-Means (k={ml['clustering']['k']}, silhueta {ml['clustering']['silhueta']}) separou "
  f"naturalmente os incidentes em perfis distintos: vazamentos antigos e pequenos; "
  f"megavazamentos recentes de baixa sensibilidade; e incidentes de <b>alta sensibilidade</b> "
  f"concentrados em saude -- os mais criticos para os titulares dos dados.")
fig_img(f_cluster, w=14, ratio=0.49)
cap("Figura 5 - Incidentes projetados em 2D (PCA) e coloridos pelo cluster atribuido.")

S.append(PageBreak())
p("<b>5.3  Deteccao de anomalias -- fora da curva</b>", "Body")
p(f"Um Isolation Forest sinalizou <b>{ml['anomalias']['total']} vazamentos atipicos</b> "
  f"(~5%) -- casos que destoam do padrao por combinarem extremos de volume, sensibilidade ou "
  f"epoca. Entre eles, megavazamentos como Shanghai Police e LinkedIn, e casos peculiares de "
  f"alta sensibilidade com baixissimo volume. Sao exatamente os incidentes que merecem "
  f"investigacao manual prioritaria.")
fig_img(f_anom, w=14, ratio=0.49)
cap("Figura 6 - Em vermelho, os incidentes sinalizados como anomalos pelo modelo.")
gap(2)
p("<b>Uso pratico:</b> modelos de anomalia funcionam como uma fila de triagem -- apontam "
  "onde a atencao humana, sempre escassa, rende mais. Comece pelos 5% mais atipicos.", "Insight")

# ============================================================
# 6. RECOMENDAÇÕES
# ============================================================
S.append(PageBreak())
h("6", "Insights e recomendacoes")
recs = [
    ("Tratar invasao como ameaca-base", f"Com {hacked_share:.0%} dos incidentes sendo 'Hacked', "
     "controles anti-intrusao (MFA, patching, deteccao continua) devem ser o piso, nao o teto."),
    ("Atacar o evitavel primeiro", "Falhas de configuracao e dispositivos perdidos respondem "
     "por parcela significativa e barata de mitigar: criptografia obrigatoria e revisao de buckets de nuvem."),
    ("Vigilancia reforcada na alta temporada", f"Concentre auditorias e simulacoes nos meses de "
     f"pico (lidera {mes_top}), sem relaxar no restante do ano."),
    ("Monitorar reincidentes de perto", f"As {n_reincid} organizacoes reincidentes precisam de "
     "pos-incidente estruturado -- vazar duas vezes indica causa-raiz nao resolvida."),
    ("Priorizar dados sensiveis (saude)", "O cluster de alta sensibilidade exige protecao "
     "reforcada: o dano por registro e muito maior, mesmo com volumes menores."),
    ("Usar anomalias como fila de triagem", "Modelos de deteccao apontam onde a investigacao "
     "humana rende mais -- comece pelos 5% mais atipicos."),
]
for i, (titulo, txt) in enumerate(recs, 1):
    S.append(Paragraph(f'<b>{i}. {titulo}.</b> {txt}', styles["Insight"]))
gap(10)
S.append(HRFlowable(width="100%", color=colors.HexColor("#bdc3c7"), thickness=.8))
gap(5)
p("<b>Metodologia.</b> Limpeza e padronizacao em Pandas; visualizacao em Plotly e Matplotlib; "
  "modelagem em scikit-learn (Random Forest para classificacao, K-Means para clustering, "
  "Isolation Forest para deteccao de anomalias). Fonte: 'World's Biggest Data Breaches & Hacks'.",
  "Cap")

# ============================================================
# Build
# ============================================================
OUT = "outputs/Relatorio_Vazamentos_de_Dados.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.7*cm, bottomMargin=1.7*cm,
                        leftMargin=2*cm, rightMargin=2*cm,
                        title="Relatorio de Vazamentos de Dados",
                        author="Analise de Dados")
doc.build(S, onFirstPage=_cover, onLaterPages=_page)

print(f"OK - relatorio PDF salvo em {OUT}")
print("Nivel 4 (refinado) concluido!")
