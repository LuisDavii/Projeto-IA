# -*- coding: utf-8 -*-

import ssl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_style('whitegrid')

print("--- Passo 1: Investigando a Cena do Crime (os Dados) ---")

zoo = fetch_ucirepo(id=111)

df_zoo = zoo.data.original.copy()
PASTA_SAIDA = Path(__file__).resolve().parent / 'dados_zoo'
PASTA_SAIDA.mkdir(exist_ok=True)
df_zoo.to_csv(PASTA_SAIDA / 'zoo_original.csv', index=False)

print("\nAs 5 primeiras amostras do dataset Zoo são:")
print(df_zoo.head())

print("\nRelatório técnico do dataset Zoo:")
df_zoo.info()

print("\nContagem de registros corrompidos (dados faltantes):")
print(df_zoo.isnull().sum())

coluna_classe = 'type' if 'type' in df_zoo.columns else 'class_type'
print("\nDistribuição das classes:")
print(df_zoo[coluna_classe].value_counts().sort_index())
print("\nProporção das classes (%):")
print((df_zoo[coluna_classe].value_counts(normalize=True).sort_index() * 100).round(2))

plt.figure(figsize=(8, 4))
sns.countplot(data=df_zoo, x=coluna_classe, order=sorted(df_zoo[coluna_classe].unique()))
plt.title('Distribuição das classes do dataset Zoo')
plt.xlabel('Classe')
plt.ylabel('Quantidade de animais')
plt.tight_layout()
plt.show()

atributos = [col for col in df_zoo.columns if col not in ['animal_name', coluna_classe]]
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for ax, atributo in zip(axes.ravel(), atributos):
    sns.countplot(data=df_zoo, x=atributo, ax=ax)
    ax.set_title(atributo)
    ax.set_xlabel('')
    ax.set_ylabel('Contagem')
for ax in axes.ravel()[len(atributos):]:
    ax.set_visible(False)
fig.suptitle('Distribuição dos atributos do dataset Zoo', y=1.02)
plt.tight_layout()
plt.show()

print("\n--- Passo 2: Consertando as Medidas Faltantes ---")

colunas_numericas = df_zoo.select_dtypes(include=[np.number]).columns

medianas = df_zoo[colunas_numericas].median()

df_zoo.fillna(medianas, inplace=True)

print("\nVerificação após o primeiro conserto (dados faltantes):")
print(df_zoo.isnull().sum())

print("\n--- Passo 3: Encontrando Outliers ---")


plt.figure(figsize=(8, 4))
sns.boxplot(x=df_zoo['legs'])
plt.title('Boxplot do Atributo legs (Número de Pernas) - ANTES do Tratamento')
plt.savefig('boxplot_legs_antes.png', bbox_inches='tight')
print("Gráfico 'boxplot_legs_antes.png' salvo com sucesso.")
plt.show()

Q1 = df_zoo['legs'].quantile(0.25)
Q3 = df_zoo['legs'].quantile(0.75)
IQR = Q3 - Q1

limite_superior = Q3 + 1.5 * IQR
limite_inferior = Q1 - 1.5 * IQR

outliers = df_zoo[(df_zoo['legs'] > limite_superior) | (df_zoo['legs'] < limite_inferior)]
outliers_count = outliers.shape[0]

print(f"\nAnálise de 'legs':")
print(f"Q1 (25%): {Q1}")
print(f"Q3 (75%): {Q3}")
print(f"IQR: {IQR}")
print(f"Limite superior para outliers: {limite_superior:.2f}")
print(f"Total de amostras identificadas como outliers: {outliers_count}")
if outliers_count > 0:
    print("Animais considerados outliers pelo método IQR:")
    print(outliers[['animal_name', 'legs']])

df_zoo_capped = df_zoo.copy()
df_zoo_capped['legs'] = np.where(
    df_zoo_capped['legs'] > limite_superior,
    limite_superior,
    df_zoo_capped['legs']
)
print("\nOutliers em 'legs' tratados com sucesso via técnica de Capping.")

print("\n--- Passo 4: Inspeção Final ---")

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_zoo_capped['legs'])
plt.title('Boxplot de legs (Número de Pernas) - APÓS o Tratamento')
plt.savefig('boxplot_legs_depois.png', bbox_inches='tight')
print("Gráfico 'boxplot_legs_depois.png' salvo com sucesso.")
plt.show()

print("\nEstatísticas descritivas do dataset Zoo limpo:")
print(df_zoo_capped.describe())

# Normalização das colunas numéricas antes da modelagem, excluindo a coluna alvo.
print("\n--- Passo 5: Normalização das Features Numéricas ---")
colunas_numericas = df_zoo_capped.select_dtypes(include=[np.number]).columns.tolist()
colunas_numericas = [col for col in colunas_numericas if col not in ['animal_name', coluna_classe]]
if colunas_numericas:
    scaler = StandardScaler()
    df_zoo_capped[colunas_numericas] = scaler.fit_transform(df_zoo_capped[colunas_numericas])
    print("Colunas numéricas normalizadas com StandardScaler:")
    print(colunas_numericas)
    print(df_zoo_capped[colunas_numericas].head())
else:
    print("Nenhuma coluna numérica para normalizar.")

# Codificação de variáveis categóricas.
print("\n--- Passo 6: Codificação das Variáveis Categóricas ---")
df_zoo_codificado = df_zoo_capped.drop(columns=['animal_name']).copy()
df_zoo_codificado[coluna_classe] = df_zoo_codificado[coluna_classe].astype('category')
df_zoo_codificado = pd.get_dummies(
    df_zoo_codificado,
    columns=[coluna_classe],
    prefix='class_type',
    dtype=int
)

print("Dataset com a variável de classe codificada:")
print(df_zoo_codificado.head())
print(f"\nDimensões após a codificação: {df_zoo_codificado.shape}")
df_zoo_codificado.to_csv(PASTA_SAIDA / 'zoo_pre_processado.csv', index=False)
print(f"\nBases original e pré-processada salvas em: {PASTA_SAIDA}")
