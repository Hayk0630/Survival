import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TARGET_COLUMN = 'In-hospital_death'
DROP_COLUMNS = [
    'Length_of_stay',
    'Survival',
    'SAPS-I',
    'SOFA',
    TARGET_COLUMN,
]


def load_dataset(data_path: str):
    df = pd.read_csv(data_path)
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(DROP_COLUMNS, axis=1)
    return X, y


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_thresholds(model, X_val, y_val, thresholds):
    val_proba = model.predict_proba(X_val)[:, 1]
    rows = []
    for thr in thresholds:
        y_pred = (val_proba >= thr).astype(int)
        metrics = compute_metrics(y_val, y_pred)
        rows.append(
            {
                'threshold': float(thr),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df = threshold_df.sort_values(
        by=['recall', 'f1', 'precision', 'threshold'],
        ascending=[False, False, False, True],
    )
    best = threshold_df.iloc[0].to_dict()
    return threshold_df, best


def build_models():
    models = {
        'LogisticRegression': Pipeline(
            [
                ('imputer', SimpleImputer(strategy='mean')),
                (
                    'model',
                    LogisticRegression(
                        class_weight='balanced',
                        solver='liblinear',
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        'RandomForest': Pipeline(
            [
                ('imputer', SimpleImputer(strategy='mean')),
                (
                    'model',
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=2,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def main(data_path: str):
    X, y = load_dataset(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    thresholds = np.arange(0.10, 0.95, 0.05)
    models = build_models()
    summary_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        val_table, best = evaluate_thresholds(model, X_val, y_val, thresholds)

        best_threshold = best['threshold']
        test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (test_proba >= best_threshold).astype(int)
        test_metrics = compute_metrics(y_test, y_test_pred)

        print('\n' + '=' * 60)
        print(f"{model_name} - Validation metrics by threshold")
        print('=' * 60)
        print(val_table.to_string(index=False))

        print('\n' + '-' * 60)
        print(f"{model_name} - Selected threshold: {best_threshold:.2f}")
        print(
            f"Validation precision: {best['precision']:.4f}, recall: {best['recall']:.4f}, f1: {best['f1']:.4f}"
        )
        print(
            f"Test precision: {test_metrics['precision']:.4f}, recall: {test_metrics['recall']:.4f}, f1: {test_metrics['f1']:.4f}"
        )

        summary_rows.append(
            {
                'model': model_name,
                'selected_threshold': best_threshold,
                'val_recall': best['recall'],
                'val_precision': best['precision'],
                'val_f1': best['f1'],
                'test_recall': test_metrics['recall'],
                'test_precision': test_metrics['precision'],
                'test_f1': test_metrics['f1'],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=['test_recall', 'test_f1', 'test_precision'], ascending=False
    )

    print('\n' + '#' * 60)
    print('FINAL MODEL COMPARISON (sorted by test recall)')
    print('#' * 60)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and compare recall-focused models for in-hospital death prediction.'
    )
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file.')
    args = parser.parse_args()
    main(args.data_path)