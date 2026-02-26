import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib

class Preprocessor:
    def __init__(self, data_path: str | None = None):
        self.data_path = Path(data_path.replace('\\', '/')) if data_path else None
        self.target_column = 'In-hospital_death'
        self.drop_columns = [
            'Length_of_stay',
            'Survival',
            'SAPS-I',
            'SOFA',
            self.target_column,
        ]
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None

    def load_dataframe(self, data_path: str | None = None) -> pd.DataFrame:
        if data_path:
            path = Path(data_path.replace('\\', '/'))
        elif self.data_path:
            path = self.data_path
        else:
            raise ValueError('data_path is required if Preprocessor was not initialized with one.')
        return pd.read_csv(path)

    def _split_features_target(self, df: pd.DataFrame):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in input dataframe.")

        y = df[self.target_column].astype(int)
        X = df.drop(self.drop_columns, axis=1)
        return X, y

    def fit(self, X: pd.DataFrame):
        self.feature_columns = list(X.columns)
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError('Preprocessor is not fitted yet. Call fit() first or load a fitted preprocessor.joblib.')

        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            raise ValueError(f'Missing required feature columns: {missing_columns}')

        X_ordered = X[self.feature_columns]
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_ordered),
            columns=self.feature_columns,
            index=X.index,
        )
        return X_imputed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def load_and_preprocess(self, data_path: str | None = None, fit: bool = True):
        df = self.load_dataframe(data_path)
        X, y = self._split_features_target(df)

        if fit:
            X_processed = self.fit_transform(X)
        else:
            X_processed = self.transform(X)

        return X_processed, y

    def save(self, path: str = 'preprocessor.joblib'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = 'preprocessor.joblib'):
        return joblib.load(path)
