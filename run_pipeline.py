import argparse
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from model import Model
from preprocessor import Preprocessor


TARGET_COLUMN = 'In-hospital_death'
DROP_COLUMNS = ['Length_of_stay', 'Survival', 'SAPS-I', 'SOFA', TARGET_COLUMN]


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class Pipeline:
    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold

    def _prepare_features(self, df: pd.DataFrame):
        columns_to_drop = [column for column in DROP_COLUMNS if column in df.columns]
        return df.drop(columns_to_drop, axis=1)

    def _train_mode(self, data_path: str):
        df = pd.read_csv(data_path)
        if TARGET_COLUMN not in df.columns:
            raise ValueError("Training mode requires label column 'In-hospital_death'.")

        y = df[TARGET_COLUMN].astype(int)
        X_raw = self._prepare_features(df)

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        main_model = Model(threshold=self.threshold)
        main_model.fit(X_train, y_train, sample_weight=sample_weight)

        val_pred_main = main_model.predict(X_val)
        metrics_rows = [
            {
                'model': 'GradientBoosting',
                **compute_metrics(y_val, val_pred_main),
            }
        ]

        lr = LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=5000,
            random_state=42,
        )
        lr.fit(X_train, y_train)
        lr_proba = lr.predict_proba(X_val)[:, 1]
        lr_pred = (lr_proba >= self.threshold).astype(int)
        metrics_rows.append({'model': 'LogisticRegression', **compute_metrics(y_val, lr_pred)})

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_val)[:, 1]
        rf_pred = (rf_proba >= self.threshold).astype(int)
        metrics_rows.append({'model': 'RandomForest', **compute_metrics(y_val, rf_pred)})

        xgb_available = True
        try:
            from xgboost import XGBClassifier
        except ImportError:
            xgb_available = False

        if xgb_available:
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            scale_pos_weight = neg_count / max(pos_count, 1)
            xgb = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            xgb_proba = xgb.predict_proba(X_val)[:, 1]
            xgb_pred = (xgb_proba >= self.threshold).astype(int)
            metrics_rows.append({'model': 'XGBoost', **compute_metrics(y_val, xgb_pred)})

        metrics_df = pd.DataFrame(metrics_rows).sort_values(by=['recall', 'precision'], ascending=False)

        preprocessor.save('preprocessor.joblib')
        main_model.save('model.joblib')
        metrics_df.to_csv('tried_models_metrics.csv', index=False)

        print('Training completed.')
        print('Saved: preprocessor.joblib, model.joblib, tried_models_metrics.csv')
        print('\nTried models metrics (validation):')
        print(metrics_df.to_string(index=False))

        return metrics_df

    def _test_mode(self, data_path: str):
        preprocessor = Preprocessor.load('preprocessor.joblib')
        model = Model.load('model.joblib')

        df = pd.read_csv(data_path)
        X_raw = self._prepare_features(df)
        X_processed = preprocessor.transform(X_raw)

        predict_probas = model.predict_proba(X_processed).tolist()
        output = {
            'predict_probas': predict_probas,
            'threshold': model.threshold,
        }

        with open('predictions.json', 'w', encoding='utf-8') as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        print('Testing completed. Saved predictions.json')
        print(f"Rows predicted: {len(predict_probas)}")
        print(f"Threshold: {model.threshold}")

        return output

    def run(self, data_path: str, test: bool = False):
        if test:
            return self._test_mode(data_path)
        return self._train_mode(data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML pipeline in training or testing mode.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to train/test csv file.')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Testing mode: load joblibs and write predictions.json.',
    )
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.run(data_path=args.data_path, test=args.test)