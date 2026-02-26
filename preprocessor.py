import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.feature_columns = None

    def fit(self, X: pd.DataFrame):
        self.feature_columns = list(X.columns)
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError('Preprocessor is not fitted. Fit it first or load preprocessor.joblib.')

        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            raise ValueError(f'Missing required feature columns: {missing_columns}')

        X_ordered = X[self.feature_columns]
        X_imputed = self.imputer.transform(X_ordered)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str = 'preprocessor.joblib'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = 'preprocessor.joblib'):
        return joblib.load(path)