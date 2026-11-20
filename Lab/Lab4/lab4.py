import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FILE_PATH = Path(__file__).resolve().with_name("credit_risk_dataset.csv")
SPLIT_ASSIGNMENTS_PATH = Path(__file__).resolve().with_name("split_assignments.csv")
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 0.5
TARGET = "loan_status"
NUMERIC_COLUMNS = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    TARGET,
)
CATEGORICAL_COLUMNS = (
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
)
NUMERIC_FEATURE_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column != TARGET
)
PREDICTOR_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_COLUMNS
EXPECTED_COLUMNS = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
SPLITS = ("train", "validation", "test")
SPLIT_SEED = 42
MODEL_TYPES = ("logistic_regression", "knn", "tree", "forest")
MODEL_NAMES = (
    "majority",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
)
MODEL_PREPROCESSORS = {
    "majority": "tree",
    "logistic_regression": "logistic_regression",
    "decision_tree": "tree",
    "random_forest": "tree",
    "knn": "knn",
}
EVALUATION_LIMITATION = (
    "This is a held-out comparison under a row-level sampling assumption, not "
    "evidence of performance in future calendar periods. The dataset has no "
    "borrower identifier to verify independence across different profiles."
)

AGE_MIN_REASONABLE = 18
AGE_MAX_REASONABLE = 100
MIN_WORKING_AGE_FOR_FLAG = 14


def load_and_summarize_data():
    raw_df = pd.read_csv(FILE_PATH)
    if set(raw_df.columns) != EXPECTED_COLUMNS or len(raw_df.columns) != 12:
        raise ValueError(
            f"Expected the 12 credit-risk columns; found {raw_df.columns.tolist()}."
        )

    duplicates_removed = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates().copy()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("category")

    if df[TARGET].isna().any():
        raise ValueError("loan_status contains missing labels; target labels are not imputed.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("loan_status must use 0 = non-default and 1 = default.")

    class_counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    class_balance = pd.DataFrame(
        {
            "count": class_counts,
            "proportion": class_counts / len(df),
        }
    )

    missing_values = df.isna().sum().rename("missing_count").to_frame()
    missing_values["missing_percent"] = missing_values["missing_count"] / len(df) * 100

    unusual_age = (df["person_age"] < AGE_MIN_REASONABLE) | (
        df["person_age"] > AGE_MAX_REASONABLE
    )
    unusual_employment = (df["person_emp_length"] < 0) | (
        df["person_emp_length"]
        > df["person_age"] - MIN_WORKING_AGE_FOR_FLAG
    )
    unusual_rows = df.loc[
        unusual_age | unusual_employment,
        ["person_age", "person_emp_length", TARGET],
    ].copy()
    unusual_rows.insert(0, "csv_row", unusual_rows.index + 2)
    unusual_rows["flags"] = [
        ", ".join(
            flag
            for flag, condition in (
                ("age outside 18-100", unusual_age.loc[index]),
                ("employment length < 0 or > age - 14", unusual_employment.loc[index]),
            )
            if condition
        )
        for index in unusual_rows.index
    ]

    print(f"CSV: {FILE_PATH}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"Rows before duplicate removal: {len(raw_df):,}")
    print(f"Exact duplicate rows removed: {duplicates_removed:,}")
    print(f"Rows after duplicate removal: {len(df):,}")
    print("\nTarget coding: 0 = non-default; 1 = default")
    print(class_balance.to_string())
    print("\nMissing values per feature after duplicate removal:")
    print(missing_values.to_string())
    print(
        "\nUnusual observations "
        "(flags only; rows are retained; CSV row numbers include the header):"
    )
    if unusual_rows.empty:
        print("None")
    else:
        print(unusual_rows.to_string(index=False))

    return df


def _profile_ids(df):
    profile_index = pd.MultiIndex.from_frame(df.loc[:, PREDICTOR_COLUMNS])
    profile_codes, _ = profile_index.factorize(sort=False)
    if (profile_codes < 0).any():
        raise ValueError("Could not assign a profile ID to every row.")
    fingerprints = pd.util.hash_pandas_object(
        df.loc[:, PREDICTOR_COLUMNS], index=False
    ).map(lambda value: f"{value:016x}")
    return profile_codes, fingerprints


def load_or_create_split_assignments(
    df,
    assignments_path=SPLIT_ASSIGNMENTS_PATH,
):
    """Load saved assignments or create a seeded, stratified grouped split."""
    if not df.index.is_unique:
        raise ValueError("DataFrame index must uniquely identify source rows.")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Split creation requires known binary target labels.")

    row_ids = pd.Index(df.index.map(str), name="row_id")
    if row_ids.has_duplicates:
        raise ValueError("Stringified source row IDs must be unique.")
    profile_codes, profile_fingerprints = _profile_ids(df)
    current_profiles = pd.DataFrame(
        {
            "row_id": row_ids,
            "profile_id": profile_fingerprints.to_numpy(),
            TARGET: df[TARGET].to_numpy(),
        }
    )

    assignments_path = Path(assignments_path)
    if assignments_path.exists():
        assignments = pd.read_csv(
            assignments_path,
            dtype={"row_id": str, "profile_id": str, "split": str},
        )
        required_columns = {"row_id", "profile_id", TARGET, "split"}
        if not required_columns.issubset(assignments.columns):
            raise ValueError(
                f"Saved assignments must contain {sorted(required_columns)}."
            )
        if assignments["row_id"].duplicated().any():
            raise ValueError("Saved split assignments contain duplicate row IDs.")

        saved = assignments.set_index("row_id")
        expected = current_profiles.set_index("row_id")
        if set(saved.index) != set(expected.index):
            raise ValueError(
                "Saved split assignments do not match the current rows. "
                "Review the data and explicitly remove the old assignment file "
                "to generate a new split."
            )
        saved = saved.loc[expected.index]
        if not saved["profile_id"].equals(expected["profile_id"]):
            raise ValueError(
                "Predictor profiles differ from the saved assignments. "
                "Review the data and explicitly remove the old assignment file "
                "to generate a new split."
            )
        if not pd.to_numeric(saved[TARGET], errors="raise").equals(expected[TARGET]):
            raise ValueError(
                "Target labels differ from the saved assignments. "
                "Review the data and explicitly remove the old assignment file "
                "to generate a new split."
            )
        if not saved["split"].isin(SPLITS).all():
            raise ValueError("Saved assignments contain an unknown split name.")
        if saved.groupby("profile_id", observed=True)["split"].nunique().gt(1).any():
            raise ValueError("A predictor profile appears in more than one split.")

        return saved.reset_index()[["row_id", "profile_id", TARGET, "split"]]

    conflicting_profiles = (
        df.assign(_profile_id=profile_codes)
        .groupby("_profile_id", observed=True)[TARGET]
        .nunique()
        .gt(1)
        .sum()
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    fold_by_row = np.full(len(df), -1, dtype=int)
    for fold_number, (_, held_out_indices) in enumerate(
        splitter.split(
            df.loc[:, PREDICTOR_COLUMNS],
            df[TARGET],
            groups=profile_codes,
        )
    ):
        fold_by_row[held_out_indices] = fold_number
    if (fold_by_row < 0).any():
        raise RuntimeError("The grouped split did not assign every row to a fold.")

    split_by_fold = {
        0: "validation",
        1: "test",
        2: "train",
        3: "train",
        4: "train",
    }
    assignments = current_profiles.copy()
    assignments["split"] = [split_by_fold[fold] for fold in fold_by_row]
    if (
        assignments.groupby("profile_id", observed=True)["split"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise RuntimeError("Grouped stratification split a predictor profile.")

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    print(
        f"\nPredictor profiles with conflicting target labels: "
        f"{conflicting_profiles:,}"
    )
    print(EVALUATION_LIMITATION)
    return assignments


def apply_saved_split(df, assignments_path=SPLIT_ASSIGNMENTS_PATH):
    """Return train, validation, and test frames using saved row assignments."""
    assignments = load_or_create_split_assignments(df, assignments_path)
    split_by_row = assignments.set_index("row_id")["split"]
    row_splits = split_by_row.loc[df.index.map(str)].to_numpy()

    split_frames = {
        split: df.iloc[np.flatnonzero(row_splits == split)].copy()
        for split in SPLITS
    }
    for split, split_df in split_frames.items():
        print(
            f"{split.capitalize():10} rows: {len(split_df):,}; "
            f"class counts: {split_df[TARGET].value_counts().sort_index().to_dict()}"
        )
    return split_frames


def _prepare_predictors(df):
    predictors = df.loc[:, PREDICTOR_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        predictors[column] = predictors[column].astype(object)
    return predictors


def build_preprocessor(model_type):
    """Create preprocessing; fit the returned transformer on training data only."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type in ("logistic_regression", "knn"):
        # StandardScaler assigns scale_ = 1 to constant features.
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="MISSING"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )


def preprocess_splits(train_df, validation_df, test_df, model_type):
    """Fit transformations on training only and apply them to all three splits."""
    preprocessor = build_preprocessor(model_type)
    X_train = preprocessor.fit_transform(_prepare_predictors(train_df))
    X_validation = preprocessor.transform(_prepare_predictors(validation_df))
    X_test = preprocessor.transform(_prepare_predictors(test_df))
    return {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": train_df[TARGET].copy(),
        "X_validation": X_validation,
        "y_validation": validation_df[TARGET].copy(),
        "X_test": X_test,
        "y_test": test_df[TARGET].copy(),
    }


def build_candidate_models():
    """Create the requested classifiers with the specified starter settings."""
    return {
        "majority": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            penalty="l2",
            C=1,
            max_iter=2_000,
            random_state=SPLIT_SEED,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=SPLIT_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=SPLIT_SEED,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=15,
            metric="euclidean",
            algorithm="brute",
            n_jobs=-1,
        ),
    }


def _positive_class_probability(model, features):
    class_index = np.flatnonzero(model.classes_ == 1)
    if len(class_index) != 1:
        raise ValueError("The fitted model does not have a positive class (1).")
    return model.predict_proba(features)[:, class_index[0]]


def _classification_metrics(labels, probabilities, threshold=DEFAULT_THRESHOLD):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision_default": precision_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "recall_default": recall_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "f1_default": f1_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(labels, probabilities),
        "brier_score": brier_score_loss(labels, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def _fit_candidate(model_name, fit_df):
    preprocessor = build_preprocessor(MODEL_PREPROCESSORS[model_name])
    X_fit = preprocessor.fit_transform(_prepare_predictors(fit_df))
    model = build_candidate_models()[model_name]
    model.fit(X_fit, fit_df[TARGET])
    return {"model": model, "preprocessor": preprocessor}


def train_and_compare_models(split_frames):
    """Fit candidates on training only and compare them on validation."""
    train_df = split_frames["train"]
    validation_df = split_frames["validation"]
    fitted_models = {}
    validation_probabilities = {}
    rows = []

    for model_name in MODEL_NAMES:
        fitted = _fit_candidate(model_name, train_df)
        X_validation = fitted["preprocessor"].transform(
            _prepare_predictors(validation_df)
        )
        probabilities = _positive_class_probability(fitted["model"], X_validation)
        validation_probabilities[model_name] = probabilities
        rows.append(
            {
                "model": model_name,
                **_classification_metrics(
                    validation_df[TARGET],
                    probabilities,
                ),
            }
        )
        fitted_models[model_name] = fitted

    validation_results = pd.DataFrame(rows).set_index("model")
    return validation_results, fitted_models, validation_probabilities


def select_by_validation_auc(validation_results):
    """Select the maximum validation AUC; exact ties follow MODEL_NAMES order."""
    missing = set(MODEL_NAMES) - set(validation_results.index)
    if missing:
        raise ValueError(f"Validation results are missing candidates: {sorted(missing)}")
    aucs = validation_results.loc[list(MODEL_NAMES), "roc_auc"]
    if aucs.isna().any():
        raise ValueError("Validation AUC values must not be missing.")
    maximum_auc = aucs.max()
    return next(
        model_name for model_name in MODEL_NAMES if aucs[model_name] == maximum_auc
    )


def _plot_roc_curves(labels, probabilities_by_model, title, output_path):
    fig, axis = plt.subplots(figsize=(8, 6))
    for model_name in probabilities_by_model:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels, probabilities_by_model[model_name]
        )
        auc = roc_auc_score(labels, probabilities_by_model[model_name])
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC = {auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.5)")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate (default recall)",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(test_results, output_path):
    fig, axes = plt.subplots(1, len(test_results), figsize=(10, 4))
    for axis, (model_name, row) in zip(axes, test_results.iterrows()):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Non-default", "Default"],
            yticklabels=["Non-default", "Default"],
            xlabel="Predicted",
            ylabel="Actual",
            title=model_name,
        )
        for (row_index, column_index), value in np.ndenumerate(matrix):
            axis.text(column_index, row_index, f"{value:,}", ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Test-set confusion matrices (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _freeze_selection(model_name, validation_results):
    model = build_candidate_models()[model_name]
    return {
        "model": model_name,
        "validation_roc_auc": float(validation_results.loc[model_name, "roc_auc"]),
        "threshold": DEFAULT_THRESHOLD,
        "feature_columns": list(PREDICTOR_COLUMNS),
        "preprocessing": {
            "numeric_imputation": "training median",
            "categorical_imputation": "MISSING",
            "categorical_encoding": "one-hot; ignore unseen categories",
            "numeric_scaling": (
                "training mean/std; constant features use scale 1"
                if MODEL_PREPROCESSORS[model_name]
                in ("logistic_regression", "knn")
                else "none"
            ),
        },
        "settings": {
            key: value
            for key, value in model.get_params(deep=False).items()
            if key in {
                "strategy",
                "penalty",
                "C",
                "max_iter",
                "max_depth",
                "min_samples_leaf",
                "n_estimators",
                "n_neighbors",
                "metric",
                "algorithm",
                "random_state",
            }
        },
    }


def evaluate_frozen_model_and_baseline(split_frames, model_name):
    """Refit the frozen candidate on train+validation, then evaluate test once."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown candidate: {model_name}")

    fit_df = pd.concat(
        [split_frames["train"], split_frames["validation"]], axis=0
    )
    test_df = split_frames["test"]
    test_predictors = _prepare_predictors(test_df)

    selected = _fit_candidate(model_name, fit_df)
    X_test = selected["preprocessor"].transform(test_predictors)
    test_probabilities = _positive_class_probability(selected["model"], X_test)

    baseline = _fit_candidate("majority", fit_df)
    X_test_baseline = baseline["preprocessor"].transform(test_predictors)
    baseline_probabilities = _positive_class_probability(
        baseline["model"], X_test_baseline
    )

    test_results = pd.DataFrame(
        [
            {
                "model": model_name,
                **_classification_metrics(test_df[TARGET], test_probabilities),
            },
            {
                "model": "majority",
                **_classification_metrics(test_df[TARGET], baseline_probabilities),
            },
        ]
    ).set_index("model")
    test_probabilities_by_model = {
        model_name: test_probabilities,
        "majority": baseline_probabilities,
    }
    return test_results, test_probabilities_by_model


def refit_selected_model_on_all_data(df, selection, output_path=None):
    """Refit the frozen candidate on all labelled rows for future predictions."""
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Final refit requires known binary target labels.")
    model_name = selection["model"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown selected candidate: {model_name}")

    fitted = _fit_candidate(model_name, df)
    artifact = {
        **fitted,
        "selection": selection,
        "threshold": selection["threshold"],
        "feature_columns": list(PREDICTOR_COLUMNS),
    }
    if output_path is not None:
        joblib.dump(artifact, output_path)
    return artifact


def predict_new_cases(new_df, model_artifact):
    """Predict default probabilities and thresholded labels for new borrowers."""
    missing_columns = set(PREDICTOR_COLUMNS) - set(new_df.columns)
    if missing_columns:
        raise ValueError(f"New cases are missing predictors: {sorted(missing_columns)}")
    X_new = model_artifact["preprocessor"].transform(
        _prepare_predictors(new_df.loc[:, PREDICTOR_COLUMNS])
    )
    probabilities = _positive_class_probability(model_artifact["model"], X_new)
    return pd.DataFrame(
        {
            "probability_default": probabilities,
            "predicted_loan_status": (
                probabilities >= model_artifact["threshold"]
            ).astype(int),
        },
        index=new_df.index,
    )


if __name__ == "__main__":
    df = load_and_summarize_data()
    split_frames = apply_saved_split(df)
    print(f"\nEvaluation limitation: {EVALUATION_LIMITATION}")
    (
        validation_results,
        validation_models,
        validation_probabilities,
    ) = train_and_compare_models(split_frames)
    selected_name = select_by_validation_auc(validation_results)
    selection = _freeze_selection(selected_name, validation_results)

    print("\nValidation comparison (test set not used for model selection):")
    print(validation_results.round(4).to_string())
    print(
        f"\nSelected by validation ROC AUC: {selected_name} "
        f"(AUC = {selection['validation_roc_auc']:.4f}); "
        f"exact ties follow candidate order {list(MODEL_NAMES)}."
    )
    validation_results.to_csv(OUTPUT_DIRECTORY / "validation_comparison.csv")
    _plot_roc_curves(
        split_frames["validation"][TARGET],
        validation_probabilities,
        "Validation ROC curves",
        OUTPUT_DIRECTORY / "validation_roc_curves.png",
    )

    test_results, test_probabilities = evaluate_frozen_model_and_baseline(
        split_frames, selected_name
    )
    print(
        "\nTest comparison (frozen validation winner refit on train + validation; "
        "baseline included):"
    )
    print(test_results.round(4).to_string())
    print("\nTest confusion matrices (rows = actual [0, 1], columns = predicted [0, 1]):")
    for model_name, row in test_results.iterrows():
        print(
            f"{model_name}:\n"
            f"[[{row['TN']}, {row['FP']}],\n"
            f" [{row['FN']}, {row['TP']}]]"
        )
    test_results.to_csv(OUTPUT_DIRECTORY / "test_comparison.csv")
    _plot_roc_curves(
        split_frames["test"][TARGET],
        test_probabilities,
        "Test ROC curves (selected candidate and majority baseline)",
        OUTPUT_DIRECTORY / "test_roc_curves.png",
    )
    _plot_confusion_matrices(
        test_results,
        OUTPUT_DIRECTORY / "test_confusion_matrices.png",
    )

    selection_path = OUTPUT_DIRECTORY / "selected_model.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    final_model = refit_selected_model_on_all_data(
        df,
        selection,
        OUTPUT_DIRECTORY / "credit_risk_final_model.joblib",
    )
    print(
        "\nAfter the one-time test evaluation, the frozen configuration was refit "
        "on all labelled rows for new-case prediction."
    )
