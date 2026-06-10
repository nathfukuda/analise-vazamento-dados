# ============================================================
# NÍVEL 1 — Enriquecimento da Análise
# 1. Sazonalidade (quais meses têm mais ataques)
# 2. Registros expostos por método de ataque
# 3. Ranking de entidades com múltiplos ataques
# ============================================================

import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Console Windows usa cp1252; força UTF-8 para emojis/acentos nos prints
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

df = pd.read_csv("data/processed/breaches_clean.csv")

# ------------------------------------------------------------
# Parsing do mês a partir da coluna `date` (ex.: "Aug 2022")
# ------------------------------------------------------------
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NAMES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}

def parse_month(val):
    if pd.isna(val):
        return np.nan
    token = str(val).strip().lower()[:3]
    return MONTHS.get(token, np.nan)

df["month"] = df["date"].apply(parse_month)

# ============================================================
# 1. SAZONALIDADE — quais meses concentram mais ataques
# ============================================================
season = (
    df.dropna(subset=["month"])
    .groupby("month")
    .agg(incidentes=("entity", "size"), registros=("records", "sum"))
    .reset_index()
)
season["mes"] = season["month"].map(MONTH_NAMES)
season = season.sort_values("month")

fig = px.bar(
    season, x="mes", y="incidentes",
    title="📅 Sazonalidade — Incidentes por Mês do Ano",
    color="incidentes", color_continuous_scale="Sunset",
    text="incidentes",
    labels={"mes": "Mês", "incidentes": "Nº de Incidentes"},
)
fig.update_traces(textposition="outside")
fig.update_layout(showlegend=False)
fig.write_html("outputs/charts/09_sazonalidade_mes.html")
print("✅ Gráfico 9 (sazonalidade) salvo")

# Heatmap mês × ano para enxergar concentração temporal
heat = (
    df.dropna(subset=["month"])
    .pivot_table(index="month", columns="year", aggfunc="size", fill_value=0)
)
heat.index = [MONTH_NAMES[m] for m in heat.index]
fig = px.imshow(
    heat, color_continuous_scale="YlOrRd", aspect="auto",
    title="🔥 Concentração de Ataques — Mês × Ano",
    labels={"x": "Ano", "y": "Mês", "color": "Incidentes"},
)
fig.write_html("outputs/charts/10_heatmap_mes_ano.html")
print("✅ Gráfico 10 (heatmap mês×ano) salvo")

top_mes = season.nlargest(3, "incidentes")[["mes", "incidentes"]]

# ============================================================
# 2. REGISTROS EXPOSTOS POR MÉTODO DE ATAQUE
# ============================================================
by_method = (
    df.groupby("method")
    .agg(
        incidentes=("entity", "size"),
        total_registros=("records", "sum"),
        mediana_registros=("records", "median"),
    )
    .reset_index()
    .sort_values("total_registros", ascending=False)
)
by_method["total_bi"] = (by_method["total_registros"] / 1e9).round(2)
by_method["mediana_mi"] = (by_method["mediana_registros"] / 1e6).round(2)
by_method["registros_por_incidente_mi"] = (
    by_method["total_registros"] / by_method["incidentes"] / 1e6
).round(2)

fig = go.Figure()
fig.add_bar(
    x=by_method["method"], y=by_method["total_bi"],
    name="Total exposto (bi)", marker_color="#e74c3c",
    text=by_method["total_bi"], textposition="outside",
)
fig.add_trace(go.Scatter(
    x=by_method["method"], y=by_method["incidentes"],
    name="Nº de incidentes", yaxis="y2",
    mode="lines+markers", marker_color="#2c3e50",
))
fig.update_layout(
    title="🔓 Volume de Registros Expostos por Método de Ataque",
    yaxis=dict(title="Registros (bilhões)"),
    yaxis2=dict(title="Nº de incidentes", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.1),
)
fig.write_html("outputs/charts/11_registros_por_metodo.html")
print("✅ Gráfico 11 (registros por método) salvo")

# ============================================================
# 3. RANKING DE ENTIDADES COM MÚLTIPLOS ATAQUES
# ============================================================
repeat = (
    df.groupby("entity")
    .agg(
        ataques=("entity", "size"),
        total_registros=("records", "sum"),
        metodos=("method", lambda s: ", ".join(sorted(s.unique()))),
        anos=("year", lambda s: ", ".join(map(str, sorted(s.unique())))),
    )
    .reset_index()
)
repeat = repeat[repeat["ataques"] > 1].sort_values(
    ["ataques", "total_registros"], ascending=False
)
repeat["total_mi"] = (repeat["total_registros"] / 1e6).round(1)

top_repeat = repeat.head(15)
fig = px.bar(
    top_repeat.sort_values("ataques"),
    x="ataques", y="entity", orientation="h",
    title="🔁 Entidades que Sofreram Múltiplos Ataques (Top 15)",
    color="total_mi", color_continuous_scale="Reds",
    text="ataques",
    labels={"ataques": "Nº de Ataques", "entity": "Entidade",
            "total_mi": "Registros (mi)"},
    hover_data={"metodos": True, "anos": True, "total_mi": True},
)
fig.update_traces(textposition="outside")
fig.update_layout(height=600)
fig.write_html("outputs/charts/12_entidades_reincidentes.html")
print("✅ Gráfico 12 (entidades reincidentes) salvo")

# ============================================================
# RESUMO EXECUTIVO DO NÍVEL 1 (salvo para reuso nos níveis 2 e 4)
# ============================================================
print("\n" + "=" * 60)
print("RESUMO — NÍVEL 1")
print("=" * 60)
print(f"\n📅 Meses com mais ataques:")
for _, r in top_mes.iterrows():
    print(f"   • {r['mes']}: {int(r['incidentes'])} incidentes")

print(f"\n🔓 Método mais destrutivo (volume exposto):")
worst = by_method.iloc[0]
print(f"   • {worst['method']}: {worst['total_bi']} bi de registros "
      f"em {int(worst['incidentes'])} incidentes")

print(f"\n🔁 Entidades reincidentes: {len(repeat)}")
if len(repeat):
    top1 = repeat.iloc[0]
    print(f"   • Líder: {top1['entity']} — {int(top1['ataques'])} ataques "
          f"({top1['anos']})")

# Persistir tabelas para os próximos níveis
season.to_csv("data/processed/n1_sazonalidade.csv", index=False)
by_method.to_csv("data/processed/n1_por_metodo.csv", index=False)
repeat.to_csv("data/processed/n1_reincidentes.csv", index=False)
print("\n✅ Tabelas do Nível 1 salvas em data/processed/")
print("🎉 Nível 1 concluído!")
