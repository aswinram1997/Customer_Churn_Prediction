import pandas as pd

class DataLoader:
    def __init__(self, file_path, sheet_name='E Comm'):
        self.file_path = file_path
        self.sheet_name = sheet_name

    def load_data(self):
        try:
            # Read the Excel file and store it as a DataFrame
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            
            # Drop CustomerID as it's not required for prediction
            df = df.drop(columns=['CustomerID'])
            
            # Success message
            print("Data loaded successfully!")

            # Return the DataFrame
            return df

        except Exception as e:
            print("Error while loading data:", e)
            return None
