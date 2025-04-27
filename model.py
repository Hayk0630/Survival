from sklearn.ensemble import RandomForestClassifier
import joblib


class Model:
    def __init__(self):
        self.model = RandomForestClassifier(bootstrap=True, class_weight='balanced', max_depth=45, min_samples_leaf=7,
                min_samples_split=41, n_estimators=35)

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'model.joblib')
        return self

    def predict(self, X_new):
        model = joblib.load('model.joblib')
        pred = model.predict(X_new)
        return pred