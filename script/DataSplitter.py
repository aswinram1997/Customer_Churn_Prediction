import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, df, target_column='Churn', train_size=0.80):
        self.df = df
        self.target_column = target_column
        self.train_size = train_size
        self.test_size = 1 - train_size

    def split_data(self):
        # Separate the features (X) and the target variable (y)
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
    
        # Success message
        print("Data splitted successfully!")

    
        return X_train, X_test, y_train, y_test

