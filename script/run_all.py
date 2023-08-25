import pandas as pd
import numpy as np
import os
from DataLoader import DataLoader
from DataExplorer import DataExplorer
from DataSplitter import DataSplitter
from DataPreprocessor import DataPreprocessor
from Modeler import Modeler
from Interpreter import Interpreter


if __name__ == "__main__":
    # Provide the file path to the Excel file
    file_path = '../data/E Commerce Dataset.xlsx'

    # Load data using data_loader
    data_loader = DataLoader(file_path, sheet_name='E Comm')
    df = data_loader.load_data()
    
    # Create an instance of the DataExplorer class with the original DataFrame
    data_explorer = DataExplorer(df)

    # Explore and save visualizations for the data
    data_explorer.explore_data()

    # Create an instance of the DataSplitter class
    data_splitter = DataSplitter(df, target_column='Churn', train_size=0.80)

    # Call the split_data method to split the data into train and test sets
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Create an instance of the DataPreprocessor class with the already split data
    data_preprocessor = DataPreprocessor(X_train, X_test, y_train, y_test)

    # Call the preprocess_data method to perform data preprocessing steps
    X_train_processed_resampled, y_train_resampled, X_test_processed, y_test = data_preprocessor.preprocess_data()
    
    # Save the preprocessing steps
    data_preprocessor.save_preprocessing_steps()
    
    # Create an instance of the Modeler class
    modeler = Modeler(X_train_processed_resampled, y_train_resampled, X_test_processed, y_test)

    # Train and evaluate the model
    train_score, test_score, y_test_pred = modeler.run_model()

    # Save the trained model
    modeler.save_model(file_name='my_model.h5')

    # Create an instance of the Interpreter class with the trained model and processed training data
    interpreter = Interpreter(modeler.model, X_train_processed_resampled)

    # Interpret the model on the test data and visualize SHAP summary plot
    interpreter.interpret(X_test_processed)
    
    # Dashboard plot generation
    # Convert y_test_pred to 1 or 0 based on threshold 0.5
    y_pred_binary = np.where(y_test_pred >= 0.5, 1, 0)
    # Create a DataFrame combining X_test and y_pred_binary
    df_combined = X_test.copy()
    df_combined['Churn Probability'] = y_test_pred.flatten()
    df_combined['Churn'] = y_pred_binary.flatten()
    # Filter the DataFrame to keep only at-risk customers (Churn Prediction = 1)
    at_risk_customers_df = df_combined[df_combined['Churn'] == 1]

    # Save churn_chart
    interpreter.churn_chart(df_combined)
    # Save tenure_chart
    interpreter.tenure_chart(at_risk_customers_df)
    # Save complain_chart
    interpreter.complain_chart(at_risk_customers_df)
    # Save cashback_chart
    interpreter.cashback_chart(at_risk_customers_df)
    # Save satisfaction_chart
    interpreter.satisfaction_chart(at_risk_customers_df)
    # Save lastorder_chart
    interpreter.lastorder_chart(at_risk_customers_df)
    # Save addresscount_chart
    interpreter.addresscount_chart(at_risk_customers_df)
    # Save ordercat_chart
    interpreter.ordercat_chart(at_risk_customers_df)
    # Save citytier_chart
    interpreter.citytier_chart(at_risk_customers_df)
