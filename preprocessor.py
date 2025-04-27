from sklearn.impute import SimpleImputer
import joblib


class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X):
        X = X.drop(['recordid', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'], axis=1)
        self.imputer.fit(X)
        joblib.dump(self.imputer, 'preprocessor.joblib')
        return self

    def transform(self, X):
        X = X.drop(['recordid', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'], axis=1)
        prep = joblib.load('preprocessor.joblib')
        X = prep.transform(X)
        return X
        