from sklearn.ensemble import GradientBoostingClassifier
import joblib


class Model:
    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
        )

    def fit(self, X, y, sample_weight=None, save_path: str | None = None):
        self.model.fit(X, y, sample_weight=sample_weight)
        if save_path:
            self.save(save_path)
        return self

    def save(self, path: str = 'model.joblib'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = 'model.joblib'):
        return joblib.load(path)

    def predict_proba(self, X_new):
        return self.model.predict_proba(X_new)[:, 1]

    def predict(self, X_new, threshold: float | None = None):
        used_threshold = self.threshold if threshold is None else threshold
        prediction_probas = self.predict_proba(X_new)
        return (prediction_probas >= used_threshold).astype(int)
