import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataExplorer:
    def __init__(self, df):
        self.df = df

    def explore_data(self):
        # Create the "images" folder if it doesn't exist in the parent directory
        images_folder_path = os.path.join(os.path.pardir, "images")
        if not os.path.exists(images_folder_path):
            os.makedirs(images_folder_path)

        # Set the desired color palette
        color_palette = ['steelblue', 'crimson']

        # Get the number of columns in the DataFrame
        num_cols = len(self.df.columns)

        # Calculate the number of rows needed to accommodate all the charts (assuming 4 columns per row)
        num_rows = (num_cols + 3) // 4

        # Create the composite figure and axes
        fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, 4 * num_rows))

        # Flatten the axes array to iterate over each subplot
        axes = axes.flatten()

        for i, column in enumerate(self.df.columns):
            # Create a new figure and axes for each plot
            ax = axes[i]

            # Check if the current column is numerical
            if self.df[column].dtype == 'float64' or self.df[column].dtype == 'int64':
                # Plot a histogram for numerical features
                sns.histplot(data=self.df, x=column, kde=True, hue='Churn', multiple='stack', palette=color_palette, ax=ax)
                ax.set_title(f'Distribution of {column}')
                ax.tick_params(axis='x')  # Rotate x-axis labels if needed
            else:
                # Plot a barplot for categorical features
                sns.countplot(data=self.df, x=column, hue='Churn', palette=color_palette, ax=ax)
                ax.set_title(f'Count of {column}')
                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if needed

        # Remove any empty subplots
        for i in range(num_cols, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout
        fig.tight_layout()

        # Save the composite plot as a PNG image in the "images" folder in the parent directory
        plt.savefig(os.path.join(images_folder_path, "dashboard.png"))

        # Close the current figure to free up memory
        plt.close()

        # Success message
        print("Data explored successfully!")
