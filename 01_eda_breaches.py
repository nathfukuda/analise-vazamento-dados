# ============================================================
# Data Breaches - Análise Exploratória Completa
# Dataset: World's Biggest Data Breaches & Hacks
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120

# ============================================================
# 1. CARREGAMENTO
# ============================================================
FILE = "data/raw/Balloon Race Data Breaches - LATEST - breaches.csv"
df = pd.read_csv(FILE, encoding="utf-8")

# Padroniza nomes das colunas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# Renomeia colunas para nomes simples
df.rename(columns={
    "organisation": "entity",
    "records_lost": "records",
    "method": "method",
    "sector": "sector",
    "year": "year"
}, inplace=True)

# ============================================================
# 2. LIMPEZA
# ============================================================

# Records: remove formatação e converte
df["records"] = (
    df["records"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.extract(r"(\d+)")[0]
)
df["records"] = pd.to_numeric(df["records"], errors="coerce")

# Year: garante inteiro válido
df["year"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
df = df[df["year"].between(2000, 2026)]

# Remove linhas sem dados essenciais
df.dropna(subset=["records", "year"], inplace=True)
df["records"] = df["records"].astype(int)
df["year"] = df["year"].astype(int)

# Padroniza texto
for col in ["method", "sector", "entity"]:
    df[col] = df[col].astype(str).str.strip().str.title()

print(f"✅ Dataset limpo: {df.shape[0]} registros, {df.shape[1]} colunas")
print(df[["entity", "year", "records", "sector", "method"]].head(10))

df.to_csv("data/processed/breaches_clean.csv", index=False)
print("✅ Salvo em data/processed/breaches_clean.csv\n")

# ============================================================
# 3. VISUALIZAÇÕES
# ============================================================

# ── 3.1 Incidentes por ano ──────────────────────────────────
incidents_year = df.groupby("year").size().reset_index(name="incidents")

fig = px.bar(
    incidents_year, x="year", y="incidents",
    title="📅 Número de Incidentes por Ano",
    color="incidents", color_continuous_scale="Reds",
    labels={"year": "Ano", "incidents": "Nº de Incidentes"}
)
fig.update_layout(showlegend=False)
fig.write_html("outputs/charts/01_incidentes_por_ano.html")
print("✅ Gráfico 1 salvo")

# ── 3.2 Registros expostos por ano ─────────────────────────
records_year = df.groupby("year")["records"].sum().reset_index()
records_year["records_bi"] = records_year["records"] / 1e9

fig = px.area(
    records_year, x="year", y="records_bi",
    title="📊 Total de Registros Expostos por Ano (bilhões)",
    labels={"year": "Ano", "records_bi": "Registros (bilhões)"},
    color_discrete_sequence=["#e74c3c"]
)
fig.write_html("outputs/charts/02_registros_por_ano.html")
print("✅ Gráfico 2 salvo")

# ── 3.3 Top 20 maiores vazamentos ──────────────────────────
top20 = df.nlargest(20, "records")[["entity", "year", "records"]].copy()
top20["records_mi"] = (top20["records"] / 1e6).round(1)

fig = px.bar(
    top20.sort_values("records"),
    x="records_mi", y="entity", orientation="h",
    title="🏆 Top 20 Maiores Vazamentos (milhões de registros)",
    color="records_mi", color_continuous_scale="OrRd",
    text="records_mi",
    labels={"records_mi": "Registros (milhões)", "entity": "Organização"}
)
fig.update_traces(texttemplate="%{text:.0f}M", textposition="outside")
fig.update_layout(showlegend=False, height=600)
fig.write_html("outputs/charts/03_top20_vazamentos.html")
print("✅ Gráfico 3 salvo")

# ── 3.4 Setores mais afetados ──────────────────────────────
sector_counts = df["sector"].value_counts().head(12).reset_index()
sector_counts.columns = ["sector", "count"]

fig = px.pie(
    sector_counts, names="sector", values="count",
    title="🏢 Setores Mais Afetados",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig.write_html("outputs/charts/04_setores.html")
print("✅ Gráfico 4 salvo")

# ── 3.5 Métodos de ataque ──────────────────────────────────
method_counts = df["method"].value_counts().head(10).reset_index()
method_counts.columns = ["method", "count"]

fig = px.bar(
    method_counts, x="count", y="method", orientation="h",
    title="🔓 Métodos de Ataque Mais Comuns",
    color="count", color_continuous_scale="Blues",
    labels={"count": "Nº de Incidentes", "method": "Método"}
)
fig.update_layout(showlegend=False)
fig.write_html("outputs/charts/05_metodos_ataque.html")
print("✅ Gráfico 5 salvo")

# ── 3.6 Heatmap setor x método ─────────────────────────────
top_sectors = df["sector"].value_counts().head(8).index
top_methods = df["method"].value_counts().head(8).index

heat_df = df[df["sector"].isin(top_sectors) & df["method"].isin(top_methods)]
heat_pivot = heat_df.pivot_table(index="sector", columns="method", aggfunc="size", fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(heat_pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
ax.set_title("🔥 Heatmap: Setor × Método de Ataque", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("outputs/charts/06_heatmap_setor_metodo.png", dpi=150)
plt.close()
print("✅ Gráfico 6 salvo")

# ── 3.7 Mediana de registros por ano ───────────────────────
avg_records = df.groupby("year")["records"].median().reset_index()
avg_records["records_mi"] = avg_records["records"] / 1e6

fig = px.line(
    avg_records, x="year", y="records_mi",
    title="📈 Mediana de Registros Expostos por Ano (milhões)",
    markers=True,
    labels={"year": "Ano", "records_mi": "Registros (milhões)"},
    color_discrete_sequence=["#e67e22"]
)
fig.write_html("outputs/charts/07_mediana_registros_ano.html")
print("✅ Gráfico 7 salvo")

# ── 3.8 Setores por volume total ───────────────────────────
sector_records = (
    df.groupby("sector")["records"].sum()
    .sort_values(ascending=False).head(10).reset_index()
)
sector_records["records_bi"] = (sector_records["records"] / 1e9).round(2)

fig = px.bar(
    sector_records, x="sector", y="records_bi",
    title="💾 Setores com Maior Volume de Dados Expostos (bilhões)",
    color="records_bi", color_continuous_scale="Reds",
    labels={"sector": "Setor", "records_bi": "Registros (bilhões)"}
)
fig.update_layout(showlegend=False)
fig.write_html("outputs/charts/08_setores_volume.html")
print("✅ Gráfico 8 salvo")

print("\n🎉 Análise concluída! Gráficos salvos em outputs/charts/")
