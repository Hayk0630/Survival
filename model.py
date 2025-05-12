from sklearn.linear_model import LogisticRegression
import joblib


class Model:
    def __init__(self):
        self.model = LogisticRegression(C=0.1, class_weight='balanced',solver='saga', max_iter=5000)

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'model.joblib')
        return self

    def predict(self, X_new):
        model = joblib.load('model.joblib')
        pred = model.predict(X_new)
        return pred
