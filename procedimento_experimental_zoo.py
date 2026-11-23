# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from src.preprocess import (
    PROJECT_ROOT,
    build_preprocessor,
    load_processed_zoo,
    preprocess_zoo,
)

sns.set_style('whitegrid')

# --- Parte 1: Carregamento da base limpa e definição de diretórios ---
PASTA_PROJETO = PROJECT_ROOT
PASTA_RESULTADOS = PASTA_PROJETO / 'resultados_zoo'
PASTA_RESULTADOS.mkdir(exist_ok=True)


def carregar_bases():
    # Recreate the clean CSV from versioned raw data; it contains no fitted scaling.
    preprocess_zoo()
    X, y = load_processed_zoo()
    df_orig = pd.read_csv(PASTA_PROJETO / 'data' / 'raw' / 'zoo.csv')
    col_classe = 'type' if 'type' in df_orig.columns else 'class_type'
    cols_atributos = X.columns.tolist()
    return df_orig, X, y, X.copy(), y.copy(), cols_atributos, col_classe


# --- Parte 2: Descrição estatística do dataset exigida nas instruções ---
def exibir_descricao_dataset(df_orig, cols_atributos, col_classe):
    # Calcula número de exemplos, atributos, classes e proporção da classe majoritária
    n_exemplos = len(df_orig)
    n_atributos = len(cols_atributos)
    contagem_classes = df_orig[col_classe].value_counts().sort_index()
    n_classes = contagem_classes.shape[0]
    classe_maj = contagem_classes.idxmax()
    n_exemplos_maj = contagem_classes.max()
    prop_maj = (n_exemplos_maj / n_exemplos) * 100

    nomes_classes = {
        1: 'Mamífero (Mammal)',
        2: 'Ave (Bird)',
        3: 'Réptil (Reptile)',
        4: 'Peixe (Fish)',
        5: 'Anfíbio (Amphibian)',
        6: 'Inseto (Bug/Insect)',
        7: 'Invertebrado (Invertebrate)',
    }

    print("=" * 75)
    print("DESCRIÇÃO E ANÁLISE DO DATASET")
    print("=" * 75)
    print("Nome do Dataset                       : Zoo (UCI Machine Learning Repository - ID 111)")
    print(f"Número total de exemplos              : {n_exemplos}")
    print(f"Número de atributos preditivos        : {n_atributos} ({', '.join(cols_atributos)})")
    print(f"Número de classes                     : {n_classes}")
    print(f"Classe majoritária                    : Classe {classe_maj} - {nomes_classes.get(classe_maj, '')} "
          f"({n_exemplos_maj} exemplos | {prop_maj:.2f}%)")
    print("\nDistribuição completa por classe:")
    df_dist = pd.DataFrame({
        'Classe': contagem_classes.index,
        'Descrição': [nomes_classes.get(c, '') for c in contagem_classes.index],
        'Nº Exemplos': contagem_classes.values,
        'Proporção (%)': ((contagem_classes.values / n_exemplos) * 100).round(2),
    })
    print(df_dist.to_string(index=False))
    df_dist.to_csv(PASTA_RESULTADOS / 'descricao_classes_zoo.csv', index=False)
    return df_dist


# --- Parte 3: Configuração dos algoritmos e das grades de parâmetros (GridSearch) ---
def obter_algoritmos_e_grades():
    # Define os 6 algoritmos do SKLearn e 4 combinações de hiperparâmetros para cada um (mínimo exigido: 3)
    return {
        'Árvore de Decisão': {
            'estimador': DecisionTreeClassifier(random_state=42),
            'grade': [
                {'criterion': ['gini'], 'max_depth': [None], 'min_samples_split': [2]},
                {'criterion': ['gini'], 'max_depth': [5], 'min_samples_split': [2]},
                {'criterion': ['entropy'], 'max_depth': [None], 'min_samples_split': [2]},
                {'criterion': ['entropy'], 'max_depth': [5], 'min_samples_split': [4]},
            ],
        },
        'KNN': {
            'estimador': KNeighborsClassifier(),
            'grade': [
                {'n_neighbors': [3], 'weights': ['uniform'], 'metric': ['euclidean']},
                {'n_neighbors': [5], 'weights': ['uniform'], 'metric': ['euclidean']},
                {'n_neighbors': [5], 'weights': ['distance'], 'metric': ['manhattan']},
                {'n_neighbors': [7], 'weights': ['distance'], 'metric': ['euclidean']},
            ],
        },
        'Naive Bayes': {
            'estimador': GaussianNB(),
            'grade': [
                {'var_smoothing': [1e-9]},
                {'var_smoothing': [1e-5]},
                {'var_smoothing': [1e-2]},
                {'var_smoothing': [1e-1]},
            ],
        },
        'Regressão Logística': {
            'estimador': LogisticRegression(random_state=42, max_iter=1000),
            'grade': [
                {'C': [0.1], 'solver': ['lbfgs']},
                {'C': [1.0], 'solver': ['lbfgs']},
                {'C': [10.0], 'solver': ['lbfgs']},
                {'C': [1.0], 'solver': ['newton-cg']},
            ],
        },
        'SVM': {
            'estimador': SVC(random_state=42),
            'grade': [
                {'C': [0.5], 'kernel': ['linear'], 'gamma': ['scale']},
                {'C': [1.0], 'kernel': ['linear'], 'gamma': ['scale']},
                {'C': [1.0], 'kernel': ['rbf'], 'gamma': ['scale']},
                {'C': [10.0], 'kernel': ['rbf'], 'gamma': ['auto']},
            ],
        },
        'MLP': {
            'estimador': MLPClassifier(random_state=42, max_iter=1000),
            'grade': [
                {'hidden_layer_sizes': [(50,)], 'activation': ['relu'], 'alpha': [0.0001]},
                {'hidden_layer_sizes': [(100,)], 'activation': ['relu'], 'alpha': [0.001]},
                {'hidden_layer_sizes': [(50, 25)], 'activation': ['relu'], 'alpha': [0.0001]},
                {'hidden_layer_sizes': [(100,)], 'activation': ['tanh'], 'alpha': [0.001]},
            ],
        },
    }


# --- Parte 4: Execução do GridSearch com 10-fold Cross-Validation Estratificado ---
def executar_experimento_gridsearch(X, y, skf, scale=True):
    # Define as métricas coletadas por fold e executa o GridSearchCV para todos os algoritmos
    scorers = {
        'acuracia': make_scorer(accuracy_score),
        'precisao_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'precisao_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    }

    algoritmos = obter_algoritmos_e_grades()
    registros_combinacoes = []
    registros_folds = []
    melhores_modelos = {}

    for nome_alg, config in algoritmos.items():
        estimator = Pipeline(
            [
                ('preprocess', build_preprocessor(X, scale=scale)),
                ('classifier', config['estimador']),
            ]
        )
        param_grid = [
            {f'classifier__{key}': values for key, values in combination.items()}
            for combination in config['grade']
        ]
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=skf,
            scoring=scorers,
            refit='acuracia',
            n_jobs=1,
        )
        grid.fit(X, y)
        res = grid.cv_results_
        melhor_idx = grid.best_index_
        melhores_modelos[nome_alg] = grid.best_estimator_

        # Registra a média e o desvio padrão dos 10 folds para cada combinação de parâmetros
        for idx, params in enumerate(res['params']):
            eh_melhor = (idx == melhor_idx)
            registros_combinacoes.append({
                'Algoritmo': nome_alg,
                'Combinação': f'Comb. {idx + 1}',
                'Parâmetros': str(params),
                'Melhor_Combinação': 'Sim' if eh_melhor else 'Não',
                'Acurácia_Média': res['mean_test_acuracia'][idx],
                'Acurácia_DP': res['std_test_acuracia'][idx],
                'Precisão_Macro_Média': res['mean_test_precisao_macro'][idx],
                'Precisão_Macro_DP': res['std_test_precisao_macro'][idx],
                'Recall_Macro_Média': res['mean_test_recall_macro'][idx],
                'Recall_Macro_DP': res['std_test_recall_macro'][idx],
                'F1_Macro_Média': res['mean_test_f1_macro'][idx],
                'F1_Macro_DP': res['std_test_f1_macro'][idx],
                'Precisão_Weighted_Média': res['mean_test_precisao_weighted'][idx],
                'Precisão_Weighted_DP': res['std_test_precisao_weighted'][idx],
                'Recall_Weighted_Média': res['mean_test_recall_weighted'][idx],
                'Recall_Weighted_DP': res['std_test_recall_weighted'][idx],
                'F1_Weighted_Média': res['mean_test_f1_weighted'][idx],
                'F1_Weighted_DP': res['std_test_f1_weighted'][idx],
            })

        # Registra os valores individuais obtidos em cada um dos 10 folds para a melhor combinação
        metricas_fold_alg = []
        for fold in range(skf.n_splits):
            linha_fold = {
                'Algoritmo': nome_alg,
                'Melhor_Parâmetro': str(res['params'][melhor_idx]),
                'Fold': f'Fold {fold + 1}',
                'Acurácia': res[f'split{fold}_test_acuracia'][melhor_idx],
                'Precisão_Macro': res[f'split{fold}_test_precisao_macro'][melhor_idx],
                'Recall_Macro': res[f'split{fold}_test_recall_macro'][melhor_idx],
                'F1_Macro': res[f'split{fold}_test_f1_macro'][melhor_idx],
                'Precisão_Weighted': res[f'split{fold}_test_precisao_weighted'][melhor_idx],
                'Recall_Weighted': res[f'split{fold}_test_recall_weighted'][melhor_idx],
                'F1_Weighted': res[f'split{fold}_test_f1_weighted'][melhor_idx],
            }
            registros_folds.append(linha_fold)
            metricas_fold_alg.append(linha_fold)

        # Calcula e adiciona as linhas de Média e Desvio Padrão ao final dos 10 folds do algoritmo
        df_temp = pd.DataFrame(metricas_fold_alg)
        cols_m = ['Acurácia', 'Precisão_Macro', 'Recall_Macro', 'F1_Macro', 'Precisão_Weighted', 'Recall_Weighted', 'F1_Weighted']
        linha_media = {'Algoritmo': nome_alg, 'Melhor_Parâmetro': str(res['params'][melhor_idx]), 'Fold': 'Média'}
        linha_dp = {'Algoritmo': nome_alg, 'Melhor_Parâmetro': str(res['params'][melhor_idx]), 'Fold': 'Desvio Padrão'}
        for col_m in cols_m:
            linha_media[col_m] = df_temp[col_m].mean()
            linha_dp[col_m] = df_temp[col_m].std(ddof=0)
        registros_folds.append(linha_media)
        registros_folds.append(linha_dp)

    df_combinacoes = pd.DataFrame(registros_combinacoes)
    df_folds = pd.DataFrame(registros_folds)
    return df_combinacoes, df_folds, melhores_modelos


# --- Parte 5: Avaliação comparativa do impacto do pré-processamento (Base Original vs. Pré-processada) ---
def comparar_impacto_preprocessamento(X, y, melhores_modelos, skf):
    # Keep each selected classifier configuration fixed while toggling only scaling.
    from sklearn.base import clone

    registros_impacto = []
    for nome_alg, modelo_base in melhores_modelos.items():
        for nome_base, scale in [
            ('Com StandardScaler', True),
            ('Sem StandardScaler', False),
        ]:
            modelo_comparacao = clone(modelo_base).set_params(
                preprocess=build_preprocessor(X, scale=scale)
            )
            accs, precs_m, recs_m, f1s_m, f1s_w = [], [], [], [], []
            for train_idx, test_idx in skf.split(X, y):
                modelo = clone(modelo_comparacao)
                modelo.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = modelo.predict(X.iloc[test_idx])
                y_true = y.iloc[test_idx]

                accs.append(accuracy_score(y_true, y_pred))
                precs_m.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
                recs_m.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
                f1s_m.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
                f1s_w.append(f1_score(y_true, y_pred, average='weighted', zero_division=0))

            registros_impacto.append({
                'Algoritmo': nome_alg,
                'Base': nome_base,
                'Acurácia_Média': np.mean(accs),
                'Acurácia_DP': np.std(accs),
                'Precisão_Macro_Média': np.mean(precs_m),
                'Precisão_Macro_DP': np.std(precs_m),
                'Recall_Macro_Média': np.mean(recs_m),
                'Recall_Macro_DP': np.std(recs_m),
                'F1_Macro_Média': np.mean(f1s_m),
                'F1_Macro_DP': np.std(f1s_m),
                'F1_Weighted_Média': np.mean(f1s_w),
                'F1_Weighted_DP': np.std(f1s_w),
            })

    return pd.DataFrame(registros_impacto)


# --- Parte 6: Geração e salvamento das tabelas de resultados ---
def salvar_e_exibir_tabelas(df_combinacoes, df_folds, df_impacto):
    # Salva os CSVs completos e gera a tabela resumo com Média ± Desvio Padrão para o artigo SBC
    df_combinacoes.to_csv(PASTA_RESULTADOS / 'resultados_gridsearch_todas_combinacoes.csv', index=False)
    df_folds.to_csv(PASTA_RESULTADOS / 'resultados_por_fold_melhores_modelos.csv', index=False)
    df_impacto.to_csv(PASTA_RESULTADOS / 'comparacao_impacto_preprocessamento.csv', index=False)

    df_melhores = df_combinacoes[df_combinacoes['Melhor_Combinação'] == 'Sim'].copy()
    df_resumo = pd.DataFrame({
        'Algoritmo': df_melhores['Algoritmo'],
        'Melhor Configuração': df_melhores['Parâmetros'],
        'Acurácia (Média ± DP)': df_melhores.apply(
            lambda r: f"{r['Acurácia_Média']*100:.2f}% ± {r['Acurácia_DP']*100:.2f}%", axis=1
        ),
        'Precisão Macro (Média ± DP)': df_melhores.apply(
            lambda r: f"{r['Precisão_Macro_Média']*100:.2f}% ± {r['Precisão_Macro_DP']*100:.2f}%", axis=1
        ),
        'Recall Macro (Média ± DP)': df_melhores.apply(
            lambda r: f"{r['Recall_Macro_Média']*100:.2f}% ± {r['Recall_Macro_DP']*100:.2f}%", axis=1
        ),
        'F1-Score Macro (Média ± DP)': df_melhores.apply(
            lambda r: f"{r['F1_Macro_Média']*100:.2f}% ± {r['F1_Macro_DP']*100:.2f}%", axis=1
        ),
        'F1-Score Weighted (Média ± DP)': df_melhores.apply(
            lambda r: f"{r['F1_Weighted_Média']*100:.2f}% ± {r['F1_Weighted_DP']*100:.2f}%", axis=1
        ),
    })
    df_resumo.to_csv(PASTA_RESULTADOS / 'resumo_comparativo_algoritmos.csv', index=False)

    print("\n" + "=" * 75)
    print("DESEMPENHO DE TODAS AS COMBINAÇÕES DE PARÂMETROS (10-FOLD CV ESTRATIFICADO)")
    print("=" * 75)
    cols_exibicao = [
        'Algoritmo', 'Combinação', 'Melhor_Combinação',
        'Acurácia_Média', 'Acurácia_DP', 'F1_Macro_Média', 'F1_Macro_DP', 'F1_Weighted_Média', 'Parâmetros'
    ]
    df_print = df_combinacoes[cols_exibicao].copy()
    for c in ['Acurácia_Média', 'Acurácia_DP', 'F1_Macro_Média', 'F1_Macro_DP', 'F1_Weighted_Média']:
        df_print[c] = (df_print[c] * 100).round(2)
    print(df_print.to_string(index=False))

    print("\n" + "=" * 75)
    print("RESUMO DOS MELHORES MODELOS POR ALGORITMO (MÉDIA ± DESVIO PADRÃO)")
    print("=" * 75)
    print(df_resumo.to_string(index=False))

    print("\n" + "=" * 75)
    print("IMPACTO DO PRÉ-PROCESSAMENTO (BASE ORIGINAL VS. PRÉ-PROCESSADA)")
    print("=" * 75)
    df_imp_print = df_impacto[['Algoritmo', 'Base', 'Acurácia_Média', 'Acurácia_DP', 'F1_Macro_Média', 'F1_Weighted_Média']].copy()
    for c in ['Acurácia_Média', 'Acurácia_DP', 'F1_Macro_Média', 'F1_Weighted_Média']:
        df_imp_print[c] = (df_imp_print[c] * 100).round(2)
    print(df_imp_print.to_string(index=False))


# --- Parte 7: Geração dos gráficos para análise e inclusão no artigo SBC ---
def gerar_graficos(df_combinacoes, df_folds, df_impacto, X_prep, y_prep, melhores_modelos, skf):
    # 1. Gráfico de barras comparando as métricas médias e desvio padrão do melhor modelo de cada algoritmo
    df_melhores = df_combinacoes[df_combinacoes['Melhor_Combinação'] == 'Sim'].copy()
    algoritmos = df_melhores['Algoritmo'].tolist()
    x = np.arange(len(algoritmos))
    largura = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    metricas = [
        ('Acurácia_Média', 'Acurácia_DP', 'Acurácia', '#1f77b4'),
        ('Precisão_Weighted_Média', 'Precisão_Weighted_DP', 'Precisão (Weighted)', '#2ca02c'),
        ('F1_Weighted_Média', 'F1_Weighted_DP', 'F1-Score (Weighted)', '#ff7f0e'),
        ('F1_Macro_Média', 'F1_Macro_DP', 'F1-Score (Macro)', '#d62728'),
    ]
    for i, (col_m, col_dp, label, cor) in enumerate(metricas):
        desloc = (i - 1.5) * largura
        valores = df_melhores[col_m].values * 100
        erros = df_melhores[col_dp].values * 100
        ax.bar(x + desloc, valores, largura, yerr=erros, capsize=3, label=label, color=cor, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos, rotation=15)
    ax.set_ylabel('Desempenho (%)')
    ax.set_ylim(50, 108)
    ax.set_title('Comparação de Desempenho dos Algoritmos (10-Fold CV Estratificado - Média ± DP)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / 'comparacao_algoritmos_metricas.png', dpi=300)
    plt.close()

    # 2. Boxplot da distribuição da acurácia nos 10 folds por algoritmo
    plt.figure(figsize=(10, 5))
    df_folds_plot = df_folds[df_folds['Fold'].str.startswith('Fold')].copy()
    df_folds_plot['Acurácia (%)'] = df_folds_plot['Acurácia'] * 100
    sns.boxplot(
        data=df_folds_plot,
        x='Algoritmo',
        y='Acurácia (%)',
        palette='Set2',
        showmeans=True,
        meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
    )
    sns.stripplot(data=df_folds_plot, x='Algoritmo', y='Acurácia (%)', color='black', alpha=0.5, jitter=0.15)
    plt.title('Distribuição da Acurácia nos 10 Folds Estratificados por Algoritmo')
    plt.ylabel('Acurácia (%)')
    plt.xlabel('Algoritmo')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / 'boxplot_acuracia_10folds.png', dpi=300)
    plt.close()

    # 3. Gráfico do desempenho por combinação de parâmetros (GridSearch) para cada algoritmo
    plt.figure(figsize=(11, 6))
    df_comb_plot = df_combinacoes.copy()
    df_comb_plot['Acurácia (%)'] = df_comb_plot['Acurácia_Média'] * 100
    sns.barplot(
        data=df_comb_plot,
        x='Algoritmo',
        y='Acurácia (%)',
        hue='Combinação',
        palette='viridis',
    )
    plt.ylim(60, 103)
    plt.title('Desempenho por Combinação de Hiperparâmetros (GridSearch - 10-Fold CV)')
    plt.ylabel('Acurácia Média (%)')
    plt.xlabel('Algoritmo')
    plt.xticks(rotation=15)
    plt.legend(title='Hiperparâmetros', loc='lower right')
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / 'desempenho_combinacoes_parametros.png', dpi=300)
    plt.close()

    # 4. Gráfico do impacto do pré-processamento (Base Original vs. Pré-processada)
    plt.figure(figsize=(10, 5.5))
    df_imp_plot = df_impacto.copy()
    df_imp_plot['Acurácia (%)'] = df_imp_plot['Acurácia_Média'] * 100
    sns.barplot(
        data=df_imp_plot,
        x='Algoritmo',
        y='Acurácia (%)',
        hue='Base',
        palette=['#e07a5f', '#3d405b'],
    )
    plt.ylim(60, 103)
    plt.title('Impacto do StandardScaler na Acurácia Média (10-Fold CV Estratificado)')
    plt.ylabel('Acurácia Média (%)')
    plt.xlabel('Algoritmo')
    plt.xticks(rotation=15)
    plt.legend(title='Conjunto de Dados', loc='lower right')
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / 'impacto_preprocessamento.png', dpi=300)
    plt.close()

    # 5. Matrizes de confusão acumuladas nos 10 folds para todos os algoritmos
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    classes_ordenadas = sorted(y_prep.unique())
    for ax, (nome_alg, modelo) in zip(axes.ravel(), melhores_modelos.items()):
        y_pred_cv = cross_val_predict(modelo, X_prep, y_prep, cv=skf)
        cm = confusion_matrix(y_prep, y_pred_cv, labels=classes_ordenadas)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes_ordenadas,
            yticklabels=classes_ordenadas,
            cbar=False,
            ax=ax,
        )
        acc_cv = accuracy_score(y_prep, y_pred_cv) * 100
        ax.set_title(f'{nome_alg} (Acc: {acc_cv:.1f}%)')
        ax.set_xlabel('Classe Predita')
        ax.set_ylabel('Classe Real')

    fig.suptitle('Matrizes de Confusão Acumuladas (10-Fold Cross-Validation Estratificado)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / 'matrizes_confusao_algoritmos.png', dpi=300, bbox_inches='tight')
    plt.close()


# --- Parte 8: Fluxo principal de execução ---
def main():
    # Orquestra carregamento, descrição do dataset, validação cruzada 10-fold, tabelas e gráficos
    df_orig, _, _, X_prep, y_prep, cols_atributos, col_classe = carregar_bases()
    exibir_descricao_dataset(df_orig, cols_atributos, col_classe)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    df_combinacoes, df_folds, melhores_modelos = executar_experimento_gridsearch(X_prep, y_prep, skf)
    df_impacto = comparar_impacto_preprocessamento(X_prep, y_prep, melhores_modelos, skf)

    salvar_e_exibir_tabelas(df_combinacoes, df_folds, df_impacto)
    gerar_graficos(df_combinacoes, df_folds, df_impacto, X_prep, y_prep, melhores_modelos, skf)

    print("\n" + "=" * 75)
    print(f"ARQUIVOS GERADOS COM SUCESSO EM: {PASTA_RESULTADOS}")
    for arq in sorted(PASTA_RESULTADOS.iterdir()):
        print(f" - {arq.name}")
    print("=" * 75)


if __name__ == '__main__':
    main()
