"""Summarize dataset size, class balance, and class distributions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocess import PROJECT_ROOT, PROCESSED_DIR, RAW_DIR, TARGET_CANDIDATES


TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CSV_PATH = TABLES_DIR / "dataset_info.csv"
TEX_PATH = TABLES_DIR / "dataset_info.tex"
FIGURE_PATH = FIGURES_DIR / "class_distribution.png"

DATASET_METADATA = {
    "zoo": {
        "name": "Zoo",
        "source_url": "https://archive.ics.uci.edu/dataset/111/zoo",
    },
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


def _find_target(columns: pd.Index, dataset_name: str, version: str) -> str:
    target = next((name for name in TARGET_CANDIDATES if name in columns), None)
    if target is None:
        raise ValueError(
            f"{dataset_name}: target column not found in {version}; "
            f"expected one of {TARGET_CANDIDATES}."
        )
    return target


def _imbalance_level(ratio: float) -> str:
    # Reporting convention: <=3 is low, <=10 is moderate, and >10 is high.
    if ratio <= 3:
        return "Baixo"
    if ratio <= 10:
        return "Moderado"
    return "Alto"


def collect_dataset_info() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Collect metadata for each raw CSV that has a matching processed CSV."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {RAW_DIR}")

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV datasets found in {RAW_DIR}")

    records = []
    distributions: dict[str, pd.Series] = {}
    for raw_path in raw_files:
        dataset_key = raw_path.stem.lower()
        metadata = DATASET_METADATA.get(dataset_key)
        if metadata is None:
            raise ValueError(
                f"No source URL metadata registered for dataset '{raw_path.stem}'. "
                "Add its name and source_url to DATASET_METADATA."
            )

        processed_path = PROCESSED_DIR / raw_path.name
        if not processed_path.is_file():
            raise FileNotFoundError(
                f"Processed dataset not found for {raw_path.name}: {processed_path}"
            )

        raw = pd.read_csv(raw_path)
        processed = pd.read_csv(processed_path)
        name = metadata["name"]
        raw_target = _find_target(raw.columns, name, "raw CSV")
        processed_target = _find_target(processed.columns, name, "processed CSV")

        class_counts = raw[raw_target].dropna().value_counts().sort_index()
        if class_counts.empty:
            raise ValueError(f"{name}: no labeled examples found in the raw CSV.")

        majority_class = class_counts.idxmax()
        majority_count = int(class_counts.max())
        minority_count = int(class_counts.min())
        labeled_examples = int(class_counts.sum())
        imbalance_ratio = majority_count / minority_count
        raw_attributes = [column for column in raw.columns if column != raw_target]
        processed_attributes = [
            column for column in processed.columns if column != processed_target
        ]
        dataset_distribution = class_counts.copy()
        if dataset_key == "zoo":
            dataset_distribution.index = [
                f"{class_id} - {CLASS_NAMES_ZOO.get(class_id, 'Classe')}"
                for class_id in dataset_distribution.index
            ]
        distributions[name] = dataset_distribution

        records.append(
            {
                "Dataset": name,
                "Fonte_URL": metadata["source_url"],
                "N_Exemplos": len(raw),
                "N_Exemplos_Processados": len(processed),
                "N_Atributos_Antes": len(raw_attributes),
                "N_Atributos_Depois": len(processed_attributes),
                "N_Classes": len(class_counts),
                "Classe_Majoritária": majority_class,
                "N_Exemplos_Classe_Majoritária": majority_count,
                "Proporção_Majoritária_pct": round(
                    majority_count / labeled_examples * 100, 2
                ),
                "Razão_Maioria_Minoria": round(imbalance_ratio, 2),
                "Nível_Desbalanceamento": _imbalance_level(imbalance_ratio),
            }
        )

    return pd.DataFrame(records), distributions


def save_class_distribution(distributions: dict[str, pd.Series]) -> None:
    """Save a count bar chart for each dataset's class distribution."""
    if not distributions:
        raise ValueError("No class distributions are available to plot.")

    figure, axes = plt.subplots(
        nrows=len(distributions),
        ncols=1,
        figsize=(10, max(4, 4 * len(distributions))),
        squeeze=False,
    )
    for axis, (dataset_name, counts) in zip(axes[:, 0], distributions.items()):
        bars = axis.bar(counts.index.astype(str), counts.values, color="#4472C4")
        axis.set_title(f"Distribuição de classes — {dataset_name}")
        axis.set_xlabel("Classe")
        axis.set_ylabel("Número de exemplos")
        axis.bar_label(bars, padding=3)
        axis.set_ylim(0, max(counts.values) * 1.15)

    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    info, distributions = collect_dataset_info()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    info.to_csv(CSV_PATH, index=False, encoding="utf-8")
    info.to_latex(TEX_PATH, index=False, escape=True, float_format="%.2f")
    save_class_distribution(distributions)
    print(f"Dataset table saved to {CSV_PATH}")
    print(f"LaTeX table saved to {TEX_PATH}")
    print(f"Class distribution figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
