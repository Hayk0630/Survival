import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, data_path: str):
        # Clean and store the path
        self.data_path = Path(data_path.replace('\\', '/'))

    def load_and_preprocess(self):
        """
        Load the CSV from the cleaned path, separate target and features,
        impute missing values with the mean strategy, and return (X, y).
        """
        # Read the dataset
        df = pd.read_csv(self.data_path)

        # Separate target variable
        y = df['In-hospital_death']

        # Drop unwanted columns
        X = df.drop([
            'Length_of_stay',
            'Survival',
            'SAPS-I',
            'SOFA',
            'In-hospital_death'
        ], axis=1)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
        
        return X_imputed, y
