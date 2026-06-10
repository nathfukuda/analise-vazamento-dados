# ============================================================
# NÍVEL 3 — Machine Learning
# 1. Classificação  : prever o MÉTODO de ataque (setor, ano, tamanho)
# 2. Clustering     : agrupar incidentes por perfil (K-Means)
# 3. Anomaly detect.: identificar vazamentos fora do padrão
# ============================================================

import sys, json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, silhouette_score)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RANDOM_STATE = 42
df = pd.read_csv("data/processed/breaches_clean.csv")

# Engenharia de atributos -------------------------------------------------
df["sector_main"] = df["sector"].str.split(",").str[0].str.strip()
df["log_records"] = np.log10(df["records"].clip(lower=1))
df["data_sensitivity"] = df["data_sensitivity"].fillna(df["data_sensitivity"].median())

results = {}  # resumo para o relatório do Nível 4

# ============================================================
# 1. CLASSIFICAÇÃO — prever o método de ataque
# ============================================================
print("=" * 60); print("1) CLASSIFICAÇÃO — método de ataque"); print("=" * 60)

feat_num = ["year", "log_records", "data_sensitivity"]
X = pd.concat([df[feat_num],
               pd.get_dummies(df["sector_main"], prefix="set")], axis=1)
y = df["method"]
feature_names = X.columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=None, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

acc = accuracy_score(y_te, pred)
# baseline: sempre prever a classe majoritária
baseline = y.value_counts(normalize=True).iloc[0]
print(f"\nAcurácia do modelo : {acc:.1%}")
print(f"Baseline (maioria) : {baseline:.1%}")
print("\n" + classification_report(y_te, pred, zero_division=0))

# Importância dos atributos
imp = (pd.Series(clf.feature_importances_, index=feature_names)
       .sort_values(ascending=False).head(12))
fig = px.bar(imp.sort_values(), orientation="h",
    title="🧠 Importância dos Atributos — Previsão do Método de Ataque",
    labels={"value": "Importância", "index": "Atributo"},
    color=imp.sort_values().values, color_continuous_scale="Viridis")
fig.update_layout(showlegend=False, template="plotly_dark", height=480)
fig.write_html("outputs/charts/13_ml_importancia.html")

# Matriz de confusão
labels = sorted(y.unique())
cm = confusion_matrix(y_te, pred, labels=labels)
fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
    color_continuous_scale="Blues",
    title="🎯 Matriz de Confusão — Método de Ataque",
    labels={"x": "Previsto", "y": "Real", "color": "Casos"})
fig.update_layout(template="plotly_dark", height=480)
fig.write_html("outputs/charts/14_ml_matriz_confusao.html")
print("✅ Gráficos 13 e 14 salvos")

results["classificacao"] = {
    "acuracia": round(acc, 3), "baseline": round(baseline, 3),
    "top_features": imp.head(5).round(3).to_dict()}

# ============================================================
# 2. CLUSTERING — agrupar incidentes por perfil (K-Means)
# ============================================================
print("\n" + "=" * 60); print("2) CLUSTERING — perfis de incidentes (K-Means)"); print("=" * 60)

clu_feats = ["year", "log_records", "data_sensitivity"]
Xc = StandardScaler().fit_transform(df[clu_feats])

# Escolhe k pela silhueta (k de 2 a 6)
sils = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xc)
    sils[k] = silhouette_score(Xc, km.labels_)
best_k = max(sils, key=sils.get)
print(f"Silhueta por k: " + ", ".join(f"k={k}:{v:.3f}" for k, v in sils.items()))
print(f"Melhor k = {best_k} (silhueta {sils[best_k]:.3f})")

km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(Xc)
df["cluster"] = km.labels_

# Perfil de cada cluster
perfil = (df.groupby("cluster")
          .agg(incidentes=("entity", "size"),
               ano_medio=("year", "mean"),
               registros_medianos=("records", "median"),
               sensibilidade_media=("data_sensitivity", "mean"),
               metodo_top=("method", lambda s: s.mode().iloc[0]),
               setor_top=("sector_main", lambda s: s.mode().iloc[0]))
          .round(1).reset_index())
print("\nPerfil dos clusters:")
print(perfil.to_string(index=False))

# Projeção PCA 2D para visualizar
pca = PCA(n_components=2, random_state=RANDOM_STATE)
proj = pca.fit_transform(Xc)
df["pc1"], df["pc2"] = proj[:, 0], proj[:, 1]
fig = px.scatter(df, x="pc1", y="pc2", color=df["cluster"].astype(str),
    size="data_sensitivity", hover_name="entity",
    hover_data={"year": True, "records": ":,", "method": True},
    title=f"🧩 Clusters de Incidentes (K-Means, k={best_k}) — projeção PCA",
    labels={"color": "Cluster", "pc1": "Componente 1", "pc2": "Componente 2"},
    color_discrete_sequence=px.colors.qualitative.Bold)
fig.update_layout(template="plotly_dark", height=520)
fig.write_html("outputs/charts/15_ml_clusters.html")
print("✅ Gráfico 15 salvo")

results["clustering"] = {"k": best_k, "silhueta": round(sils[best_k], 3),
                         "perfis": perfil.to_dict(orient="records")}

# ============================================================
# 3. ANOMALY DETECTION — vazamentos fora do padrão
# ============================================================
print("\n" + "=" * 60); print("3) ANOMALY DETECTION — Isolation Forest"); print("=" * 60)

Xa = StandardScaler().fit_transform(df[["year", "log_records", "data_sensitivity"]])
iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_estimators=300)
df["anomaly"] = iso.fit_predict(Xa)            # -1 = anomalia
df["anomaly_score"] = iso.decision_function(Xa)  # menor = mais anômalo

anom = (df[df["anomaly"] == -1]
        .sort_values("anomaly_score")
        [["entity", "year", "records", "sector_main", "method",
          "data_sensitivity", "anomaly_score"]])
print(f"\nAnomalias detectadas: {len(anom)} de {len(df)} ({len(anom)/len(df):.1%})")
print("\nTop 10 vazamentos mais atípicos:")
print(anom.head(10).to_string(index=False))

df["tipo"] = np.where(df["anomaly"] == -1, "Anomalia", "Normal")
fig = px.scatter(df, x="year", y="log_records", color="tipo",
    size="data_sensitivity", hover_name="entity",
    hover_data={"records": ":,", "method": True},
    title="🚨 Detecção de Anomalias — Vazamentos Fora do Padrão",
    labels={"log_records": "log10(registros)", "year": "Ano", "tipo": ""},
    color_discrete_map={"Anomalia": "#e74c3c", "Normal": "#3498db"})
fig.update_layout(template="plotly_dark", height=520)
fig.write_html("outputs/charts/16_ml_anomalias.html")
print("✅ Gráfico 16 salvo")

anom.to_csv("data/processed/n3_anomalias.csv", index=False)
results["anomalias"] = {
    "total": int(len(anom)),
    "lista": anom.head(10)[["entity", "year", "records", "method"]]
              .to_dict(orient="records")}

# Persistir resumo + dataset enriquecido para o Nível 4
with open("data/processed/n3_ml_resultados.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
df.to_csv("data/processed/breaches_ml.csv", index=False)

print("\n✅ Resultados de ML salvos em data/processed/")
print("🎉 Nível 3 concluído!")
