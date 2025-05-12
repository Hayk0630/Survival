import getting_the_data
import model
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import argparse
from sklearn.metrics import classification_report
class Pipeline:
    def __init__(self, data_path: str, test: bool):
        self.data_path = data_path
        self.test = test
        self.model = model.Model()
        self.preprocessor = getting_the_data.Preprocessor(data_path)

    def run(self):
        if self.test:
            preprocessor = joblib.load('preprocessor.joblib')
            self.model = joblib.load('model.joblib')
            X, y = preprocessor.load_and_preprocess()
            threshold = 0.45
            prediction_probas = self.model.predict_proba(X)[:, 1]
            predictions = (prediction_probas >= threshold).astype(int)
            return  classification_report(y, predictions)
        else:
            X, y = self.preprocessor.load_and_preprocess()
            joblib.dump(self.preprocessor, 'preprocessor.joblib')

            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.model.fit(X_train, y_train)

            prediction_probas = self.model.model.predict_proba(X_valid)[:, 1]
            threshold = 0.45
            predictions = (prediction_probas >= threshold).astype(int)

            return {
                'train_size': len(X_train),
                'valid_size': len(X_valid),
                'accuracy': (predictions == y_valid).mean(),
                'classification_report': classification_report(y_valid,predictions )
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML pipeline.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--test',required = True, help='do you want test or train ?')
    args = parser.parse_args()

    pipeline = Pipeline(data_path=args.data_path, test=args.test)
    results = pipeline.run()
    print(results)