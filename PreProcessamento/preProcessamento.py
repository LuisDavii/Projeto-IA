import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# 1. Carregamento
# ---------------------------------------------------------------------------
# O dataset 541 NÃO está disponível via ucimlrepo (fetch_ucirepo falha),
# então baixamos o zip direto do repositório da UCI e lemos os 3 CSVs.
import io
import zipfile
import urllib.request
from pathlib import Path

URL_ZIP = ("https://archive.ics.uci.edu/static/public/541/"
           "malware+static+and+dynamic+features+vxheaven+and+virus+total.zip")
PASTA = Path("dados_malware")
ARQUIVOS = {
    "benigno": "staDynBenignLab.csv",
    "vxheaven": "staDynVxHeaven2698Lab.csv",
    "virustotal": "staDynVt2955Lab.csv",
}

PASTA.mkdir(exist_ok=True)
if not all((PASTA / f).exists() for f in ARQUIVOS.values()):
    print("Baixando o dataset da UCI...")
    try:
        req = urllib.request.Request(URL_ZIP, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(PASTA)
    except Exception as e:
        raise SystemExit(
            f"Falha no download ({e}).\n"
            f"Baixe o zip manualmente em {URL_ZIP}\n"
            f"e extraia os 3 CSVs dentro da pasta '{PASTA}'."
        )

# Os CSVs podem ter sido extraídos em subpastas; procura pelo nome
def achar(nome):
    achados = list(PASTA.rglob(nome))
    if not achados:
        raise FileNotFoundError(f"{nome} não encontrado em {PASTA.resolve()}")
    return achados[0]

partes = {}
for origem, nome in ARQUIVOS.items():
    d = pd.read_csv(achar(nome))
    d.columns = d.columns.str.strip()
    partes[origem] = d
    print(f"{origem:11s}: {d.shape[0]} linhas x {d.shape[1]} colunas")

# Colunas que não existem em todos os arquivos (ex.: rótulo extra) ficam
# fora das features para evitar vazamento de dados.
comuns = set.intersection(*(set(d.columns) for d in partes.values()))
extras = set.union(*(set(d.columns) for d in partes.values())) - comuns
if extras:
    print("Colunas que não aparecem em todos os arquivos (excluídas):", sorted(extras))

ordem = [c for c in partes["virustotal"].columns if c in comuns]
df = pd.concat(
    [d[ordem].assign(origem=o) for o, d in partes.items()],
    ignore_index=True,
)

# Alvo binário: 0 = benigno, 1 = malware (VxHeaven + Virus Total)
df["malware"] = (df["origem"] != "benigno").astype(int)
target = "malware"

print(f"\nDimensões: {df.shape}")
print(df.iloc[:5, :8])  # são ~1087 colunas, mostramos só as primeiras

df.info(verbose=False)

# Separa colunas de features numéricas e não numéricas
feat_cols = [c for c in df.columns if c not in (target, 'origem')]
num_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()
nao_num = [c for c in feat_cols if c not in num_cols]
print(f"\nFeatures numéricas: {len(num_cols)} | não numéricas: {len(nao_num)}")
if nao_num:
    print("Colunas não numéricas (mantidas sem tratamento):", nao_num[:10])

# Distribuição das classes
print("\nDistribuição do alvo:")
print(df[target].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x=df[target].astype(str))
plt.title('Distribuição das classes')
plt.show()

# ---------------------------------------------------------------------------
# 2. Valores faltantes (a UCI informa que não há, mas mantemos a checagem)
# ---------------------------------------------------------------------------
faltantes = df.isnull().sum()
print(f"\nColunas com valores faltantes: {(faltantes > 0).sum()}")

medianas = df[num_cols].median()
df[num_cols] = df[num_cols].fillna(medianas)
print("Faltantes restantes nas features numéricas:", int(df[num_cols].isnull().sum().sum()))

# Remove duplicatas exatas, se houver
antes = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Linhas duplicadas removidas: {antes - len(df)}")

# ---------------------------------------------------------------------------
# 3. Outliers (IQR) - versão vetorizada para todas as features
# ---------------------------------------------------------------------------
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

# Em dados esparsos/binários o IQR costuma ser 0; aplicar capping nessas
# colunas "achataria" a feature inteira, então elas são ignoradas.
tratar = [c for c in num_cols if IQR[c] > 0 and df[c].nunique() > 2]
print(f"\nFeatures elegíveis ao tratamento de outliers: {len(tratar)} de {len(num_cols)}")

fora = ((df[tratar] < lim_inf[tratar]) | (df[tratar] > lim_sup[tratar])).sum()
print("Top 10 features com mais outliers:")
print(fora.sort_values(ascending=False).head(10))

# Plota as 4 features com mais outliers
top4 = fora.sort_values(ascending=False).head(4).index.tolist()
df_antes = df[top4].copy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Boxplots - ANTES do Tratamento')
for ax, col in zip(axes.ravel(), top4):
    sns.boxplot(ax=ax, x=df_antes[col])
    ax.set_title(col)
plt.tight_layout()
plt.show()

# Capping (winsorização pelos limites do IQR)
df[tratar] = df[tratar].clip(lower=lim_inf[tratar], upper=lim_sup[tratar], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Boxplots - APÓS o Tratamento (Capping)')
for ax, col in zip(axes.ravel(), top4):
    sns.boxplot(ax=ax, x=df[col])
    ax.set_title(col)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 4. Resumo
# ---------------------------------------------------------------------------
print(df[num_cols].describe().T.head(15))