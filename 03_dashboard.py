# ============================================================
# NÍVEL 2 — Dashboard Interativo Unificado (HTML único)
# Reúne todos os gráficos em uma única página apresentável,
# com KPIs, tema escuro e Plotly interativo (1 só plotly.js).
# ============================================================

import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
import warnings
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pio.templates.default = "plotly_dark"
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color="#e6e6e6"), margin=dict(l=40, r=20, t=40, b=40),
              autosize=True, width=None)

df = pd.read_csv("data/processed/breaches_clean.csv")

MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
          "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
MNAME = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
         7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
df["month"] = df["date"].apply(
    lambda v: MONTHS.get(str(v).strip().lower()[:3], np.nan) if pd.notna(v) else np.nan)

# ---------- KPIs ----------
total_inc = len(df)
total_rec = df["records"].sum()
n_setores = df["sector"].nunique()
reincid = (df.groupby("entity").size() > 1).sum()
periodo = f"{df.year.min()}–{df.year.max()}"

figs = []  # (titulo, fig)

# 1. Incidentes por ano
g = df.groupby("year").size().reset_index(name="inc")
figs.append(("Incidentes por Ano", px.bar(g, x="year", y="inc",
    color="inc", color_continuous_scale="Reds").update_layout(showlegend=False)))

# 2. Registros expostos por ano
g = df.groupby("year")["records"].sum().reset_index()
g["bi"] = g["records"]/1e9
figs.append(("Registros Expostos por Ano (bi)", px.area(g, x="year", y="bi",
    color_discrete_sequence=["#e74c3c"])))

# 3. Top 15 vazamentos
g = df.nlargest(15, "records")[["entity","records"]].copy()
g["mi"] = (g["records"]/1e6).round(1)
figs.append(("Top 15 Maiores Vazamentos (mi)", px.bar(g.sort_values("records"),
    x="mi", y="entity", orientation="h", color="mi",
    color_continuous_scale="OrRd").update_layout(showlegend=False)))

# 4. Setores mais afetados (rosca)
g = df["sector"].value_counts().head(10).reset_index()
g.columns = ["sector","count"]
figs.append(("Setores Mais Afetados", px.pie(g, names="sector", values="count",
    hole=0.45, color_discrete_sequence=px.colors.qualitative.Set3)))

# 5. Métodos de ataque
g = df["method"].value_counts().reset_index()
g.columns = ["method","count"]
figs.append(("Métodos de Ataque", px.bar(g, x="count", y="method",
    orientation="h", color="count",
    color_continuous_scale="Blues").update_layout(showlegend=False)))

# 6. Heatmap setor × método (agora interativo)
ts = df["sector"].value_counts().head(8).index
tm = df["method"].value_counts().head(6).index
pv = df[df.sector.isin(ts) & df.method.isin(tm)].pivot_table(
    index="sector", columns="method", aggfunc="size", fill_value=0)
figs.append(("Heatmap Setor × Método", px.imshow(pv, text_auto=True,
    color_continuous_scale="YlOrRd", aspect="auto")))

# 7. Sazonalidade
g = df.dropna(subset=["month"]).groupby("month").size().reset_index(name="inc")
g["mes"] = g["month"].map(MNAME)
g = g.sort_values("month")
figs.append(("Sazonalidade — Ataques por Mês", px.bar(g, x="mes", y="inc",
    color="inc", color_continuous_scale="Sunset",
    text="inc").update_layout(showlegend=False)))

# 8. Registros por método (volume)
g = df.groupby("method")["records"].sum().reset_index()
g["bi"] = (g["records"]/1e9).round(2)
g = g.sort_values("bi")
figs.append(("Volume Exposto por Método (bi)", px.bar(g, x="bi", y="method",
    orientation="h", color="bi", text="bi",
    color_continuous_scale="Reds").update_layout(showlegend=False)))

# 9. Entidades reincidentes
g = df.groupby("entity").size().reset_index(name="ataques")
g = g[g.ataques > 1].sort_values("ataques").tail(12)
figs.append(("Entidades Reincidentes (Top 12)", px.bar(g, x="ataques", y="entity",
    orientation="h", color="ataques", text="ataques",
    color_continuous_scale="Purples").update_layout(showlegend=False)))

# 10. Sensibilidade dos dados
g = df["data_sensitivity"].value_counts().sort_index().reset_index()
g.columns = ["sens","count"]
SENS = {1:"1 · Email/senha",2:"2 · SSN/dados pessoais",3:"3 · Cartão/financeiro",
        4:"4 · Saúde/credenciais",5:"5 · Total/comprometido"}
g["label"] = g["sens"].map(SENS).fillna(g["sens"].astype(str))
figs.append(("Sensibilidade dos Dados Vazados", px.bar(g, x="label", y="count",
    color="sens", color_continuous_scale="Turbo").update_layout(showlegend=False)))

# Aplica layout-base a todas
for _, f in figs:
    f.update_layout(**LAYOUT, height=420)

# ---------- Monta HTML ----------
cards = []
for i, (titulo, f) in enumerate(figs):
    div = f.to_html(full_html=False, include_plotlyjs=False,
                    default_width="100%", default_height="420px",
                    config={"displayModeBar": False, "responsive": True})
    cards.append(f'<div class="card"><h3>{titulo}</h3>{div}</div>')

# Plotly.js embutido inline -> arquivo único 100% offline (não depende de CDN)
plotlyjs = poff.get_plotlyjs()

KPI = lambda v, l: f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>'
kpis = "".join([
    KPI(f"{total_inc}", "Incidentes"),
    KPI(f"{total_rec/1e9:.1f} bi", "Registros expostos"),
    KPI(f"{n_setores}", "Setores"),
    KPI(f"{reincid}", "Entidades reincidentes"),
    KPI(periodo, "Período coberto"),
])

html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard · Maiores Vazamentos de Dados do Mundo</title>
<script type="text/javascript">{plotlyjs}</script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#0d1117;
        color:#e6e6e6; padding: 28px; }}
header {{ text-align:center; margin-bottom: 24px; }}
header h1 {{ font-size: 2rem; background: linear-gradient(90deg,#e74c3c,#f39c12);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
header p {{ color:#8b949e; margin-top:6px; }}
.kpis {{ display:flex; gap:16px; flex-wrap:wrap; justify-content:center;
        margin-bottom: 28px; }}
.kpi {{ background:#161b22; border:1px solid #30363d; border-radius:14px;
        padding:18px 26px; min-width:160px; text-align:center; }}
.kpi-v {{ font-size:1.8rem; font-weight:700; color:#f39c12; }}
.kpi-l {{ font-size:.8rem; color:#8b949e; margin-top:4px; text-transform:uppercase;
        letter-spacing:.5px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(440px,1fr));
        gap:20px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:14px;
        padding:16px; overflow:hidden; min-width:0; }}
.card h3 {{ font-size:1.05rem; margin-bottom:8px; color:#e6e6e6; }}
.card .plotly-graph-div {{ width:100% !important; }}
.card .js-plotly-plot, .card .plot-container {{ width:100% !important; }}
footer {{ text-align:center; color:#6e7681; margin-top:32px; font-size:.85rem; }}
</style></head>
<body>
<header>
  <h1>🛡️ Maiores Vazamentos de Dados do Mundo</h1>
  <p>Dashboard analítico interativo · {periodo} · {total_inc} incidentes analisados</p>
</header>
<div class="kpis">{kpis}</div>
<div class="grid">{''.join(cards)}</div>
<footer>Gerado com Python · Pandas · Plotly — análise exploratória de {total_inc} incidentes</footer>
</body></html>"""

OUT = "outputs/dashboard.html"
with open(OUT, "w", encoding="utf-8") as fp:
    fp.write(html)

print(f"✅ Dashboard unificado salvo em {OUT}")
print(f"   {len(figs)} gráficos · {total_inc} incidentes · {total_rec/1e9:.1f} bi registros")
print("🎉 Nível 2 concluído!")
