import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import joblib

class DataPreprocessor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessing_folder = '../model'
        self.preprocessing_file = 'preprocessing_steps.pkl'
        self.num_imputer = None
        self.scaler = None
        self.encoder = None

    def _data_cleaning(self):
        # Separate numerical and categorical features
        X_train_numerical = self.X_train.select_dtypes(include=['float64', 'int64'])
        X_train_categorical = self.X_train.select_dtypes(include=['object'])
        X_test_numerical = self.X_test.select_dtypes(include=['float64', 'int64'])
        X_test_categorical = self.X_test.select_dtypes(include=['object'])

        # Create an instance of SimpleImputer for numerical features
        num_imputer = SimpleImputer(strategy='mean')
        self.num_imputer = num_imputer

        # Fit-transform on the training set
        X_train_imputed = pd.DataFrame(num_imputer.fit_transform(X_train_numerical), columns=X_train_numerical.columns, index=X_train_numerical.index)

        # Transform on the test set
        X_test_imputed = pd.DataFrame(num_imputer.transform(X_test_numerical), columns=X_test_numerical.columns, index=X_test_numerical.index)
        
        # Success message
        print("Data cleaned successfully!")

        return X_train_imputed, X_test_imputed, num_imputer

    def _feature_scaling(self, X_train_imputed, X_test_imputed):
        # Create a StandardScaler object
        scaler = StandardScaler()
        self.scaler = scaler

        # Fit the scaler on X_train_imputed
        scaler.fit(X_train_imputed)

        # Transform X_train_imputed and X_test_imputed using the fitted scaler
        X_train_scaled = scaler.transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Convert to DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_imputed.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_imputed.columns)
        
        # Success message
        print("Data scaled successfully!")
        
        return X_train_scaled_df, X_test_scaled_df, scaler

    def _feature_encoding(self, X_train_categorical, X_test_categorical):
        # Create an instance of OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder = encoder

        # Fit the encoder on X_train_categorical
        encoder.fit(X_train_categorical)

        # Transform X_train_categorical and X_test_categorical using the fitted encoder
        X_train_encoded = encoder.transform(X_train_categorical)
        X_test_encoded = encoder.transform(X_test_categorical)

        # Convert the encoded arrays back to DataFrames
        X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(X_train_categorical.columns))
        X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(X_test_categorical.columns))

        # Success message
        print("Data encoded successfully!")
        
        return X_train_encoded_df, X_test_encoded_df, encoder
    

    def _outlier_removal(self, X_train_processed):
        # Create an instance of IsolationForest
        isolation_forest = IsolationForest(contamination=0.025, random_state=42)

        # Fit the model on your data
        isolation_forest.fit(X_train_processed)

        # Predict outliers for the training data
        outliers_train = isolation_forest.predict(X_train_processed)

        # Remove outliers from the training set
        X_train_processed_outliers_removed = X_train_processed[outliers_train == 1]
        y_train_outliers_removed = self.y_train[outliers_train == 1]
        
        # Success message
        print("Outliers removed successfully!")
        
        return X_train_processed_outliers_removed, y_train_outliers_removed

    def _handle_imbalanced_dataset(self, X_train_processed_outliers_removed, y_train_outliers_removed):
        # Create an instance of SMOTE
        smote = SMOTE(sampling_strategy=0.7)  # Generate synthetic samples for the minority class to achieve a 1:2 ratio

        # Apply SMOTE to the training data
        X_train_processed_resampled, y_train_resampled = smote.fit_resample(X_train_processed_outliers_removed, y_train_outliers_removed)

        # Count the occurrences of each class in the resampled dataset
        value_counts_resampled = y_train_resampled.value_counts()
        
        # Success message
        print("Data balanced successfully!")
        
        return X_train_processed_resampled, y_train_resampled
    
    
    def save_preprocessing_steps(self):
        # Create the folder if it does not exist
        if not os.path.exists(self.preprocessing_folder):
            os.makedirs(self.preprocessing_folder)

        # Save the preprocessing steps using joblib
        preprocessing_steps = {
            'num_imputer': self.num_imputer,
            'scaler': self.scaler,
            'encoder': self.encoder,
        }

        joblib.dump(preprocessing_steps, os.path.join(self.preprocessing_folder, self.preprocessing_file))
        print("Preprocessing steps saved successfully!")
    

    def preprocess_data(self):
        # imputation
        X_train_imputed, X_test_imputed, num_imputer = self._data_cleaning()
        # feature scaling
        X_train_scaled_df, X_test_scaled_df, scaler = self._feature_scaling(X_train_imputed, X_test_imputed)
        # feature encoding
        X_train_encoded_df, X_test_encoded_df, encoder = self._feature_encoding(self.X_train.select_dtypes(include=['object']), self.X_test.select_dtypes(include=['object']))
        # concatenate dataframes
        X_train_processed = pd.concat([X_train_scaled_df, X_train_encoded_df], axis=1)
        X_test_processed = pd.concat([X_test_scaled_df, X_test_encoded_df], axis=1)
        # outlier removal
        X_train_processed_outliers_removed, y_train_outliers_removed = self._outlier_removal(X_train_processed)
        # imbalanced data handling
        X_train_processed_resampled, y_train_resampled = self._handle_imbalanced_dataset(X_train_processed_outliers_removed, y_train_outliers_removed)
        
        # Success message
        print("Data preprocessed successfully!")
        
        return X_train_processed_resampled, y_train_resampled, X_test_processed, self.y_test


