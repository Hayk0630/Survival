import preprocessor, model
import pandas as pd
import joblib
import json
import argparse


class Pipeline:
    def __init__(self, data_path, test=True):
        self.data_path = data_path
        self.test = test
        self.model = model.Model()
        self.preprocessor = preprocessor.Preprocessor()
        self.data = None

    def run(self):
        if self.test:
            self.preprocessor = joblib.load('preprocessor.joblib')
            self.model = joblib.load('model.joblib')
            data = pd.read_csv(self.data_path)
            data = data.drop(['recordid', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'], axis=1)
            X_test = self.preprocessor.transform(data)
            pred = self.model.predict(X_test)
            threshold = 0.5
            predictions = {'predictions': pred.tolist(), 'threshold': threshold}
            with open('predictions.json', 'w') as f:
                json.dump(predictions, f)

        else:
            data = pd.read_csv(self.data_path)
            y = data['In-hospital_death']
            self.preprocessor.fit(data)
            self.data = self.preprocessor.transform(data)
            self.model.fit(self.data, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Absolute or relative path to a train or test dataset.")
    parser.add_argument("--test", action="store_true",
                        help="Run in testing mode. If nota provided, it is training mode.")
    args, unknown_args = parser.parse_known_args()

    args = parser.parse_args()

    data = pd.read_csv(args.data_path)

    pipeline = Pipeline(data_path=args.data_path, test=args.test)
    pipeline.data = data
    pipeline.run()


if __name__ == "__main__":
    main()
