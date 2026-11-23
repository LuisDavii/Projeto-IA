"""Generate publication-ready figures and LaTeX result tables."""

import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocess import RESULTS_DIR


TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_DPI = 400
METRICS = {
    "Acurácia": ("Acurácia_Média", "Acurácia_DP"),
    "F1-macro": ("F1_Macro_Média", "F1_Macro_DP"),
}
CLASS_NAMES_ZOO = {
    1: "Mamífero",
    2: "Ave",
    3: "Réptil",
    4: "Peixe",
    5: "Anfíbio",
    6: "Inseto",
    7: "Invertebrado",
}

sns.set_theme(
    style="whitegrid",
    rc={
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 15,
    },
)


def _read_table(filename: str) -> pd.DataFrame:
    path = TABLES_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python procedimento_experimental_zoo.py` first."
        )
    return pd.read_csv(path)


def _require_columns(data: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {', '.join(sorted(missing))}. "
            "Regenerate experiment outputs before plotting."
        )


def _save_figure(figure: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / filename
    figure.savefig(output.with_suffix(".png"), dpi=OUTPUT_DPI, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _load_inputs() -> dict[str, pd.DataFrame]:
    tables = {
        "folds": _read_table("resultados_por_fold_melhores_modelos.csv"),
        "grid": _read_table("resultados_gridsearch_todas_combinacoes.csv"),
        "impact": _read_table("comparacao_impacto_preprocessamento.csv"),
        "datasets": _read_table("dataset_info.csv"),
        "confusion": _read_table("matriz_confusao_oof.csv"),
    }
    _require_columns(
        tables["folds"],
        {"Dataset", "Algoritmo", "Fold", "Acurácia", "F1_Macro"},
        "resultados_por_fold_melhores_modelos.csv",
    )
    _require_columns(
        tables["grid"],
        {
            "Dataset",
            "Algoritmo",
            "Parâmetros",
            "Melhor_Combinação",
            "Acurácia_Média",
            "Acurácia_DP",
            "F1_Macro_Média",
            "F1_Macro_DP",
            "Tempo_Treino_Média_Segundos",
            "Tempo_Treino_DP_Segundos",
        },
        "resultados_gridsearch_todas_combinacoes.csv",
    )
    _require_columns(
        tables["impact"],
        {"Dataset", "Algoritmo", "Base", "Acurácia_Média", "Acurácia_DP", "F1_Macro_Média", "F1_Macro_DP"},
        "comparacao_impacto_preprocessamento.csv",
    )
    _require_columns(
        tables["datasets"],
        {"Dataset"},
        "dataset_info.csv",
    )
    _require_columns(
        tables["confusion"],
        {"Dataset", "Melhor_Algoritmo", "Classe_Real", "Classe_Predita", "Contagem"},
        "matriz_confusao_oof.csv",
    )
    return tables


def plot_fold_distributions(folds: pd.DataFrame, datasets: list[str]) -> None:
    fold_rows = folds[folds["Fold"].astype(str).str.match(r"^Fold \d+$")].copy()
    if fold_rows.empty:
        raise ValueError("No per-fold metric rows found in results tables.")

    figure, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(14, max(5, 4.5 * len(datasets))),
        squeeze=False,
    )
    for row_index, dataset in enumerate(datasets):
        subset = fold_rows[fold_rows["Dataset"] == dataset]
        for column_index, (metric, label) in enumerate(
            [("F1_Macro", "F1-macro"), ("Acurácia", "Acurácia")]
        ):
            axis = axes[row_index, column_index]
            sns.boxplot(
                data=subset,
                x="Algoritmo",
                y=metric,
                order=sorted(subset["Algoritmo"].unique()),
                color="#7EA6E0",
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                },
                ax=axis,
            )
            sns.stripplot(
                data=subset,
                x="Algoritmo",
                y=metric,
                order=sorted(subset["Algoritmo"].unique()),
                color="#263238",
                alpha=0.55,
                jitter=0.14,
                size=3.5,
                ax=axis,
            )
            axis.set_title(f"{label} por fold — {dataset}")
            axis.set_xlabel("Algoritmo")
            axis.set_ylabel(label)
            axis.set_ylim(0, 1.05)
            axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    _save_figure(figure, "boxplot_f1_accuracy_por_algoritmo")


def plot_best_model_scores(best: pd.DataFrame, datasets: list[str]) -> None:
    figure, axes = plt.subplots(
        len(datasets),
        len(METRICS),
        figsize=(14, max(5, 4.5 * len(datasets))),
        squeeze=False,
    )
    for row_index, dataset in enumerate(datasets):
        subset = best[best["Dataset"] == dataset].sort_values("Algoritmo")
        for column_index, (label, (mean_column, std_column)) in enumerate(METRICS.items()):
            axis = axes[row_index, column_index]
            values = subset[mean_column].to_numpy() * 100
            errors = subset[std_column].to_numpy() * 100
            positions = np.arange(len(subset))
            axis.bar(
                positions,
                values,
                yerr=errors,
                capsize=4,
                color="#4472C4",
                alpha=0.9,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(subset["Algoritmo"], rotation=20, ha="right")
            axis.set_ylim(0, 105)
            axis.set_ylabel(f"{label} (%)")
            axis.set_title(f"{label} — {dataset}")
            axis.grid(axis="x", visible=False)
    figure.tight_layout()
    _save_figure(figure, "medias_dp_melhores_modelos")


def _parse_parameters(value: str) -> dict[str, object]:
    parameters = ast.literal_eval(value)
    return {
        key.split("__")[-1]: parameter_value
        for key, parameter_value in parameters.items()
    }


def _format_parameters(value: str) -> str:
    parameters = _parse_parameters(value)
    return ", ".join(f"{name}={parameter!r}" for name, parameter in parameters.items())


def plot_hyperparameter_effect(grid: pd.DataFrame, datasets: list[str]) -> None:
    figure, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(13, max(4.5, 4.5 * len(datasets))),
        squeeze=False,
    )
    parameter_rows = []
    for _, row in grid.iterrows():
        parsed = _parse_parameters(row["Parâmetros"])
        parameter_rows.append(
            {
                "Dataset": row["Dataset"],
                "Algoritmo": row["Algoritmo"],
                "F1_Macro_Média": row["F1_Macro_Média"],
                **parsed,
            }
        )
    parameters = pd.DataFrame(parameter_rows)

    for row_index, dataset in enumerate(datasets):
        subset = parameters[parameters["Dataset"] == dataset]
        svc_axis, knn_axis = axes[row_index]

        svc = subset[(subset["Algoritmo"] == "SVM") & subset["C"].notna()]
        if not svc.empty:
            for kernel, kernel_rows in svc.groupby("kernel"):
                kernel_rows = kernel_rows.sort_values("C")
                svc_axis.plot(
                    kernel_rows["C"],
                    kernel_rows["F1_Macro_Média"] * 100,
                    marker="o",
                    linewidth=2,
                    label=f"Kernel {kernel}",
                )
            svc_axis.set_xlabel("C")
            svc_axis.set_ylabel("F1-macro médio (%)")
            svc_axis.set_title(f"Efeito de C no SVM — {dataset}")
            svc_axis.legend(title="Configuração")
            svc_axis.set_ylim(0, 105)
        else:
            svc_axis.text(0.5, 0.5, "Sem combinações SVM", ha="center", va="center")

        knn = subset[
            (subset["Algoritmo"] == "KNN") & subset["n_neighbors"].notna()
        ].copy()
        if not knn.empty:
            knn["Configuração"] = knn.apply(
                lambda row: f"{row['weights']}, {row['metric']}", axis=1
            )
            for configuration, configuration_rows in knn.groupby("Configuração"):
                configuration_rows = configuration_rows.sort_values("n_neighbors")
                knn_axis.plot(
                    configuration_rows["n_neighbors"],
                    configuration_rows["F1_Macro_Média"] * 100,
                    marker="o",
                    linewidth=2,
                    label=configuration,
                )
            knn_axis.set_xlabel("Número de vizinhos (k)")
            knn_axis.set_ylabel("F1-macro médio (%)")
            knn_axis.set_title(f"Efeito de k no KNN — {dataset}")
            knn_axis.legend(title="Pesos, métrica")
            knn_axis.set_ylim(0, 105)
        else:
            knn_axis.text(0.5, 0.5, "Sem combinações KNN", ha="center", va="center")

    figure.tight_layout()
    _save_figure(figure, "efeito_hiperparametros")


def plot_confusion_matrices(
    confusion: pd.DataFrame,
    datasets: list[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(6, 6 * len(datasets)), 5.5),
        squeeze=False,
    )
    for axis, dataset in zip(axes[0], datasets):
        subset = confusion[confusion["Dataset"] == dataset]
        if subset.empty:
            raise ValueError(f"No out-of-fold confusion matrix found for {dataset}.")
        algorithm = subset["Melhor_Algoritmo"].iloc[0]
        matrix = subset.pivot(
            index="Classe_Real",
            columns="Classe_Predita",
            values="Contagem",
        ).fillna(0)
        matrix = matrix.sort_index().sort_index(axis=1)
        if dataset.lower() == "zoo":
            matrix.index = [
                CLASS_NAMES_ZOO.get(int(value), str(value)) for value in matrix.index
            ]
            matrix.columns = [
                CLASS_NAMES_ZOO.get(int(value), str(value))
                for value in matrix.columns
            ]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=axis,
        )
        axis.set_title(f"{dataset} — melhor modelo: {algorithm}")
        axis.set_xlabel("Classe predita")
        axis.set_ylabel("Classe real")
    figure.tight_layout()
    _save_figure(figure, "matriz_confusao_melhor_modelo")


def plot_preprocessing_impact(impact: pd.DataFrame, datasets: list[str]) -> None:
    figure, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(14, max(5, 4.5 * len(datasets))),
        squeeze=False,
    )
    bases = ["Com StandardScaler", "Sem StandardScaler"]
    colors = ["#4472C4", "#ED7D31"]
    for row_index, dataset in enumerate(datasets):
        subset = impact[impact["Dataset"] == dataset]
        for column_index, (metric, (mean_column, std_column)) in enumerate(
            METRICS.items()
        ):
            axis = axes[row_index, column_index]
            algorithms = sorted(subset["Algoritmo"].unique())
            positions = np.arange(len(algorithms))
            width = 0.36
            for base_index, base in enumerate(bases):
                base_rows = (
                    subset[subset["Base"] == base]
                    .set_index("Algoritmo")
                    .reindex(algorithms)
                )
                offset = (base_index - 0.5) * width
                axis.bar(
                    positions + offset,
                    base_rows[mean_column].to_numpy() * 100,
                    width,
                    yerr=base_rows[std_column].to_numpy() * 100,
                    capsize=3,
                    color=colors[base_index],
                    label=base,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(algorithms, rotation=20, ha="right")
            axis.set_ylim(0, 105)
            axis.set_ylabel(f"{metric} (%)")
            axis.set_title(f"{metric} — {dataset}")
            axis.legend(title="Pré-processamento")
    figure.suptitle("Efeito do escalonamento no desempenho")
    figure.tight_layout()
    figure.subplots_adjust(top=0.9)
    _save_figure(figure, "comparacao_com_sem_preprocessamento")


def plot_training_times(best: pd.DataFrame, datasets: list[str]) -> None:
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(7, 7 * len(datasets)), 5.5),
        squeeze=False,
    )
    for axis, dataset in zip(axes[0], datasets):
        subset = best[best["Dataset"] == dataset].sort_values("Algoritmo")
        positions = np.arange(len(subset))
        bars = axis.bar(
            positions,
            subset["Tempo_Treino_Média_Segundos"],
            yerr=subset["Tempo_Treino_DP_Segundos"],
            capsize=4,
            color="#70AD47",
        )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axis.set_xticks(positions)
        axis.set_xticklabels(subset["Algoritmo"], rotation=20, ha="right")
        axis.set_ylabel("Tempo médio de ajuste por fold (s)")
        axis.set_title(f"Tempo de treino — {dataset}")
    figure.tight_layout()
    _save_figure(figure, "tempo_treino_por_algoritmo")


def _mean_std_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def write_latex_tables(
    best: pd.DataFrame,
    grid: pd.DataFrame,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    algorithm_table = best[
        [
            "Dataset",
            "Algoritmo",
            "Parâmetros",
            "Acurácia_Média",
            "Acurácia_DP",
            "F1_Macro_Média",
            "F1_Macro_DP",
            "F1_Weighted_Média",
            "F1_Weighted_DP",
        ]
    ].copy()
    algorithm_table["Acurácia (média ± DP, %)"] = algorithm_table.apply(
        lambda row: _mean_std_percent(row["Acurácia_Média"], row["Acurácia_DP"]),
        axis=1,
    )
    algorithm_table["F1-macro (média ± DP, %)"] = algorithm_table.apply(
        lambda row: _mean_std_percent(row["F1_Macro_Média"], row["F1_Macro_DP"]),
        axis=1,
    )
    algorithm_table["F1-weighted (média ± DP, %)"] = algorithm_table.apply(
        lambda row: _mean_std_percent(row["F1_Weighted_Média"], row["F1_Weighted_DP"]),
        axis=1,
    )
    algorithm_table["Parâmetros"] = algorithm_table["Parâmetros"].map(
        _format_parameters
    )
    algorithm_table = algorithm_table[
        [
            "Dataset",
            "Algoritmo",
            "Parâmetros",
            "Acurácia (média ± DP, %)",
            "F1-macro (média ± DP, %)",
            "F1-weighted (média ± DP, %)",
        ]
    ]
    algorithm_table.to_latex(
        TABLES_DIR / "resultados_por_algoritmo.tex",
        index=False,
        escape=True,
        longtable=True,
    )

    combinations = grid[
        [
            "Dataset",
            "Algoritmo",
            "Combinação",
            "Parâmetros",
            "Melhor_Combinação",
            "Acurácia_Média",
            "Acurácia_DP",
            "F1_Macro_Média",
            "F1_Macro_DP",
            "Tempo_Treino_Média_Segundos",
            "Tempo_Treino_DP_Segundos",
        ]
    ].copy()
    combinations["Acurácia (média ± DP, %)"] = combinations.apply(
        lambda row: _mean_std_percent(row["Acurácia_Média"], row["Acurácia_DP"]),
        axis=1,
    )
    combinations["F1-macro (média ± DP, %)"] = combinations.apply(
        lambda row: _mean_std_percent(row["F1_Macro_Média"], row["F1_Macro_DP"]),
        axis=1,
    )
    combinations["Tempo de treino (média ± DP, s)"] = combinations.apply(
        lambda row: (
            f"{row['Tempo_Treino_Média_Segundos']:.4f} ± "
            f"{row['Tempo_Treino_DP_Segundos']:.4f}"
        ),
        axis=1,
    )
    combinations["Parâmetros"] = combinations["Parâmetros"].map(_format_parameters)
    combinations = combinations[
        [
            "Dataset",
            "Algoritmo",
            "Combinação",
            "Parâmetros",
            "Melhor_Combinação",
            "Acurácia (média ± DP, %)",
            "F1-macro (média ± DP, %)",
            "Tempo de treino (média ± DP, s)",
        ]
    ]
    combinations.to_latex(
        TABLES_DIR / "todas_combinacoes.tex",
        index=False,
        escape=True,
        longtable=True,
    )

    # This aggregate describes the spread between each dataset's algorithm means.
    overall = best.groupby("Dataset", as_index=False).agg(
        N_Algoritmos=("Algoritmo", "nunique"),
        Acurácia_Média=("Acurácia_Média", "mean"),
        Acurácia_DP=("Acurácia_Média", lambda values: values.std(ddof=0)),
        F1_Macro_Média=("F1_Macro_Média", "mean"),
        F1_Macro_DP=("F1_Macro_Média", lambda values: values.std(ddof=0)),
    )
    overall["Acurácia média entre algoritmos (média ± DP, %)"] = overall.apply(
        lambda row: _mean_std_percent(row["Acurácia_Média"], row["Acurácia_DP"]),
        axis=1,
    )
    overall["F1-macro médio entre algoritmos (média ± DP, %)"] = overall.apply(
        lambda row: _mean_std_percent(row["F1_Macro_Média"], row["F1_Macro_DP"]),
        axis=1,
    )
    overall = overall[
        [
            "Dataset",
            "N_Algoritmos",
            "Acurácia média entre algoritmos (média ± DP, %)",
            "F1-macro médio entre algoritmos (média ± DP, %)",
        ]
    ]
    overall.to_csv(TABLES_DIR / "media_geral_por_dataset.csv", index=False)
    overall.to_latex(
        TABLES_DIR / "media_geral_por_dataset.tex",
        index=False,
        escape=True,
    )


def main() -> None:
    tables = _load_inputs()
    datasets = tables["datasets"]["Dataset"].astype(str).tolist()
    _require_columns(
        tables["grid"],
        {"Combinação"},
        "resultados_gridsearch_todas_combinacoes.csv",
    )
    best = tables["grid"][
        tables["grid"]["Melhor_Combinação"].astype(str).str.lower().isin(
            {"sim", "true", "1"}
        )
    ].copy()
    if best.empty:
        raise ValueError("No best hyperparameter combinations are marked in GridSearch output.")

    plot_fold_distributions(tables["folds"], datasets)
    plot_best_model_scores(best, datasets)
    plot_hyperparameter_effect(tables["grid"], datasets)
    plot_confusion_matrices(tables["confusion"], datasets)
    plot_preprocessing_impact(tables["impact"], datasets)
    plot_training_times(best, datasets)
    write_latex_tables(best, tables["grid"])
    print(f"PNG/PDF figures saved to {FIGURES_DIR}")
    print(f"LaTeX tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
