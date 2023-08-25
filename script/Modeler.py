import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class Modeler:
    def __init__(self, X_train_processed_resampled, y_train_resampled, X_test_processed, y_test):
        self.X_train = X_train_processed_resampled
        self.y_train = y_train_resampled
        self.X_test = X_test_processed
        self.y_test = y_test
        self.model = None

    def _build_model(self, input_shape):
        # Define the neural network architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def run_model(self, epochs=30, batch_size=12, verbose=1):
        # Build the model
        self._build_model(input_shape=(self.X_train.shape[1],))

        # Compile the model with the appropriate loss function and optimizer
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        # Train the model on the training set
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Predict probabilities for train and test sets
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Calculate ROC AUC scores for train and test sets
        DNN_train_score = roc_auc_score(self.y_train, y_train_pred)
        DNN_test_score = roc_auc_score(self.y_test, y_test_pred)
        
        # Success message
        print("DNN model trained successfully!")

        return DNN_train_score, DNN_test_score, y_test_pred
    
    def save_model(self, file_name='model.h5', folder_path='../model'):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Save the trained model to the specified folder and file name
        save_path = os.path.join(folder_path, file_name)
        self.model.save(save_path)
        print(f"Model saved successfully at: {save_path}")