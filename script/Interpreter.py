import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import seaborn as sns

class Interpreter:
    def __init__(self, model, X_train_processed_resampled):
        self.model = model
        self.X_train_processed = X_train_processed_resampled

        
    def interpret(self, X_test_processed, num_background_samples=1000):
        # Convert X_train_processed_resampled to a NumPy array
        X_train_array = np.array(self.X_train_processed)

        # Convert X_test_processed to a NumPy array
        X_test_array = np.array(X_test_processed)

        # Set a random state for reproducibility (choose any seed value you like)
        random_state = 42
        np.random.seed(random_state)

        # Choose the number of background samples you want to use
        num_background_samples = 1000

        # Randomly select num_background_samples from X_train_array
        random_indices = np.random.choice(X_train_array.shape[0], num_background_samples, replace=False)
        background_samples = X_train_array[random_indices]

        # Initialize the SHAP explainer with the model and the reduced background samples
        explainer = shap.DeepExplainer(self.model, background_samples)

        # Compute the SHAP values for the test set
        shap_values = explainer.shap_values(X_test_array)

        # Create a summary plot for the entire test set (without the legend)
        shap.summary_plot(shap_values, X_test_array, plot_type="bar", feature_names=X_test_processed.columns, show=False)

        # Get the current figure and axes
        fig, ax = plt.gcf(), plt.gca()

        # Remove the legend
        ax.get_legend().remove()

        # Set the x-axis label as "Mean Shap Value"
        plt.xlabel("Mean Shap Value")

        # Save the plot as an image in the "image" folder in the root directory
        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "shap_plot.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')

        # Success message
        print(f"SHAP summary plot saved successfully at: {image_path}")
        
        
    def churn_chart(self, df_combined):
        # Calculate the count of churn and not churn
        churn_count = sum(1 for val in df_combined['Churn'] if val == 1)
        not_churn_count = sum(1 for val in df_combined['Churn'] if val == 0)

        # Labels for the pie chart
        labels = ['Churn', 'Not Churn']

        # Sizes of the pie slices
        sizes = [churn_count, not_churn_count]

        # Colors from the coolwarm color palette
        colors = plt.cm.coolwarm([0.8, 0.2])

        # Create the pie chart
        plt.figure(figsize=(8, 6))  # Increase the figure size for HD plot
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})

        # Equal aspect ratio ensures that the pie chart is circular
        plt.axis('equal')

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "churn_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"churn_chart saved successfully at: {image_path}")
        
        
    def tenure_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='Tenure', bins=6, kde=True, color='skyblue')

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "tenure_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"tenure_chart saved successfully at: {image_path}")
        
        
    def complain_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='Complain', bins=2)

        # Colors from the coolwarm color palette with specific shades
        colors = [plt.cm.coolwarm(0.2), plt.cm.coolwarm(0.8)]

        # Apply custom colors to the bars
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        # Set x-axis ticks explicitly to show only 0 and 1 with a slight offset to center them in the bars
        plt.xticks([0.25, 0.75], ['No', 'Yes'])

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "complain_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"complain_chart saved successfully at: {image_path}")
        

    def cashback_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='CashbackAmount', bins=5, kde=True, color='skyblue')

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "cashback_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"cashback_chart saved successfully at: {image_path}")
        
        
    def satisfaction_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='SatisfactionScore', bins=5, kde=True, color='skyblue')

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        # Set x-axis ticks explicitly to show only 0 and 1 with a slight offset to center them in the bars
        plt.xticks([1.4, 2.2, 3, 3.8, 4.6],[1,2,3,4,5])

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "satisfaction_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"satisfaction_chart saved successfully at: {image_path}")  
        
        
    def lastorder_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='DaySinceLastOrder', bins=5, kde=True, color='skyblue')

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "lastorder_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"lastorder_chart saved successfully at: {image_path}")  
        
        
    def addresscount_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='NumberOfAddress', bins=10, kde=True, color='skyblue')

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors) # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        # Set the number of desired x-axis ticks (e.g., 6 ticks)
        num_ticks = 6
        plt.locator_params(axis='x', nbins=num_ticks)

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "addresscount_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"addresscount_chart saved successfully at: {image_path}")  
        
        
    def ordercat_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='PreferedOrderCat', bins=6)

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        # Set the number of desired x-axis ticks (e.g., 6 ticks)
        num_ticks = 6
        plt.xticks(rotation=45)  # Rotate x-axis ticks by 45 degrees

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "ordercat_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ordercat_chart saved successfully at: {image_path}")          
        
        
    def citytier_chart(self, at_risk_customers_df):
        # Set the style for better aesthetics
        sns.set(style='white')

        # Create the tenure distribution plot
        plt.figure(figsize=(8, 3))  # Increase the figure size for HD plot
        ax = sns.histplot(data=at_risk_customers_df, x='CityTier', bins=3)

        # Define custom colors for the bars using a color gradient (cool to warm)
        colors = sns.color_palette("coolwarm", len(ax.patches))
        colors = list(colors)  # Reverse the colors to invert the gradient
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')  # Set black edges for the bars

        # Remove the right and top spines and hide the axes
        sns.despine(right=True, top=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show axis numbers with light grey ticks
        ax.tick_params(axis='both', which='both', length=0, color='lightgrey', labelcolor='grey')

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Show count on each bar with a darker color
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

        # Add a horizontal line
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.5)

        # Set the x-axis ticks at the middle of each bar
        x_ticks_pos = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
        x_tick_labels = ["Tier 1", "Tier 2", "Tier 3"]
        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(x_tick_labels)
        plt.xticks(rotation=45)  # Rotate x-axis ticks by 45 degrees

        image_folder = "../images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "citytier_chart.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"citytier_chart saved successfully at: {image_path}")          
        
        
        
        
        
        
        
        