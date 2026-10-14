# -*- coding: utf-8 -*-

import ssl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_style('whitegrid')

print("--- Passo 1: Investigando a Cena do Crime (os Dados) ---")

zoo = fetch_ucirepo(id=111)

df_zoo = zoo.data.original.copy()

print("\nAs 5 primeiras amostras do dataset Zoo são:")
print(df_zoo.head())

print("\nRelatório técnico do dataset Zoo:")
df_zoo.info()

print("\nContagem de registros corrompidos (dados faltantes):")
print(df_zoo.isnull().sum())

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

# Codificação de variáveis categóricas.
print("\n--- Passo 5: Codificação das Variáveis Categóricas ---")

# animal_name identifica cada animal; não é uma característica para o modelo.
# class_type representa classes sem ordem numérica, então usamos One-Hot Encoding.
df_zoo_codificado = df_zoo_capped.drop(columns=['animal_name']).copy()
df_zoo_codificado['class_type'] = df_zoo_codificado['class_type'].astype('category')
df_zoo_codificado = pd.get_dummies(
    df_zoo_codificado,
    columns=['class_type'],
    prefix='class_type',
    dtype=int
)

print("Dataset com a variável de classe codificada:")
print(df_zoo_codificado.head())
print(f"\nDimensões após a codificação: {df_zoo_codificado.shape}")
