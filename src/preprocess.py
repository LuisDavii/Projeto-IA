"""Prepare project datasets and build fold-safe preprocessing pipelines."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

ZOO_RAW_PATH = RAW_DIR / "zoo.csv"
ZOO_PROCESSED_PATH = PROCESSED_DIR / "zoo.csv"
PREPROCESS_LOG_PATH = RESULTS_DIR / "preprocess_log.md"
TARGET_CANDIDATES = ("type", "class_type")
IDENTIFIER_COLUMNS = {"animal_name", "animal_id", "id"}


def preprocess_zoo(
    raw_path: Path = ZOO_RAW_PATH,
    processed_path: Path = ZOO_PROCESSED_PATH,
    log_path: Path = PREPROCESS_LOG_PATH,
) -> pd.DataFrame:
    """Clean the Zoo table without fitting data-dependent transforms globally."""
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Zoo raw CSV not found at {raw_path}. Place it at data/raw/zoo.csv."
        )

    original = pd.read_csv(raw_path)
    target = next((name for name in TARGET_CANDIDATES if name in original.columns), None)
    if target is None:
        raise ValueError(
            f"Zoo target column is missing; expected one of {TARGET_CANDIDATES}."
        )

    before_rows, before_columns = original.shape
    before_nulls = int(original.isna().sum().sum())
    missing_target_rows = int(original[target].isna().sum())
    data = original.dropna(subset=[target]).copy()

    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)

    # animal_name identifies each animal but is not a predictive measurement.
    identifiers = [
        column
        for column in data.columns
        if column != target and column.strip().lower() in IDENTIFIER_COLUMNS
    ]
    data = data.drop(columns=identifiers)

    # Constant predictors cannot distinguish examples and add no model information.
    constant_columns = [
        column
        for column in data.columns
        if column != target and data[column].nunique(dropna=False) <= 1
    ]
    data = data.drop(columns=constant_columns)

    # Zoo predictors are already numeric; any remaining categorical predictors are
    # one-hot encoded by the fold-fitted ColumnTransformer used by the experiment.
    feature_columns = [column for column in data.columns if column != target]
    feature_nulls = int(data[feature_columns].isna().sum().sum())
    if data[feature_columns].select_dtypes(exclude=np.number).shape[1] == 0:
        categorical_columns: list[str] = []
    else:
        categorical_columns = data[feature_columns].select_dtypes(
            exclude=np.number
        ).columns.tolist()

    # The binary traits and discrete "legs" values are valid Zoo domain values;
    # IQR clipping would alter legitimate animals, so no rows are removed/capped.
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_columns = identifiers + constant_columns
    log_path.write_text(
        "\n".join(
            [
                "# Preprocessing log",
                "",
                "## Zoo",
                "",
                f"- Arquivo de entrada: `{raw_path.relative_to(PROJECT_ROOT).as_posix()}`",
                f"- Arquivo tratado: `{processed_path.relative_to(PROJECT_ROOT).as_posix()}`",
                f"- Linhas antes/depois: {before_rows}/{len(data)}",
                f"- Colunas antes/depois: {before_columns}/{data.shape[1]}",
                f"- Nulos na entrada: {before_nulls}",
                f"- Nulos no CSV após a limpeza: {int(data.isna().sum().sum())}",
                f"- Linhas removidas por alvo ausente: {missing_target_rows}",
                f"- Linhas duplicadas removidas: {duplicate_rows}",
                f"- Nulos em atributos preservados para imputação no Pipeline: {feature_nulls}",
                "- Nulos imputados fora do Pipeline: 0 (imputação deve ser ajustada em cada fold).",
                f"- Colunas descartadas: {', '.join(dropped_columns) if dropped_columns else 'nenhuma'}.",
                "- Outliers: nenhum valor foi removido/capado; os atributos são binários "
                "ou discretos e os valores observados de `legs` são válidos no domínio Zoo.",
                f"- Atributos categóricos restantes: "
                f"{', '.join(categorical_columns) if categorical_columns else 'nenhum'}; "
                "OneHotEncoder é aplicado dentro do Pipeline.",
                "- StandardScaler e SimpleImputer são ajustados somente nos dados de "
                "treino de cada fold.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data


def load_processed_zoo(
    processed_path: Path = ZOO_PROCESSED_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load cleaned features and labels for model selection."""
    processed_path = Path(processed_path)
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed Zoo CSV not found at {processed_path}. "
            "Run `python -m src.preprocess` first."
        )

    data = pd.read_csv(processed_path)
    target = next((name for name in TARGET_CANDIDATES if name in data.columns), None)
    if target is None:
        raise ValueError(
            f"Processed Zoo target column is missing; expected one of {TARGET_CANDIDATES}."
        )
    return data.drop(columns=[target]), data[target].astype(int)


def build_preprocessor(features: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Create transforms that are fitted separately inside each CV training fold."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("At least one predictor column is required.")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def main() -> None:
    data = preprocess_zoo()
    print(
        f"Zoo preparado: {len(data)} linhas, {data.shape[1]} colunas. "
        f"Log: {PREPROCESS_LOG_PATH}"
    )


if __name__ == "__main__":
    main()
