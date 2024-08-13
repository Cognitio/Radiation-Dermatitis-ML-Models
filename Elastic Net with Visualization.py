import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn.ensemble import GradientBoostingRegressor

def load_data(file_path):
    return pd.read_excel(file_path)

def select_features(df):
    X = df.iloc[:, 1:-1]  # Features (excluding the first column and the outcome column)
    y = df.iloc[:, -1]    # Outcome (last column)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    
    # Perform feature selection using ElasticNetCV
    elastic_net = ElasticNetCV(cv=5, random_state=0).fit(X_train, y_train)
    importance = np.abs(elastic_net.coef_)
    
    # Get the indices of the features that are selected and excluded
    selected_indices = np.where(importance > 0)[0]
    excluded_indices = np.where(importance == 0)[0]
    
    # Create a DataFrame with the selected features
    selected_features = df.columns[selected_indices + 1]  # +1 to account for the original column indexing
    selected_df = df[selected_features]
    
    # Create a list of excluded features with reasons
    excluded_features = []
    for idx in excluded_indices:
        feature_name = df.columns[idx + 1]
        reason = "Coefficient is zero, indicating no impact on the outcome"
        excluded_features.append((feature_name, reason))
    
    return selected_df, excluded_features, elastic_net.coef_, df.columns[1:-1]

def plot_coefficients(coef, feature_names):
    # Filter out the zero coefficients
    non_zero_indices = np.where(coef != 0)[0]
    non_zero_coef = coef[non_zero_indices]
    non_zero_features = feature_names[non_zero_indices]
    
    # Plot the coefficients as a bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(non_zero_features, non_zero_coef, color='#20a7db')
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Rotate the x labels for better readability
    plt.xticks(rotation=45, fontsize = 12)
    
    # Add labels and title
    plt.xlabel('Feature', fontsize = 14)
    plt.ylabel('Coefficient Value', fontsize = 14)
    plt.title('Elastic Net Non-Zero Coefficients', fontsize = 16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_to_excel(df, excluded_features, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Selected Features', index=False)
        
        # Create a new sheet for excluded features
        excluded_df = pd.DataFrame(excluded_features, columns=['Excluded Feature', 'Reason'])
        excluded_df.to_excel(writer, sheet_name='Excluded Features', index=False)

def main(input_file, output_file):
    # Load data
    df = load_data(input_file)
    
    # Select features
    selected_df, excluded_features, coef, feature_names = select_features(df)
    
    # Save the selected features and excluded features to a new Excel file
    save_to_excel(selected_df, excluded_features, output_file)
    print(f"Selected features and excluded features saved to {output_file}")
    
    # Print excluded features
    print("\nExcluded features:")
    for feature, reason in excluded_features:
        print(f"- {feature}: {reason}")
    
    # Plot non-zero coefficients
    plot_coefficients(coef, feature_names)

if __name__ == "__main__":
    input_file = ""  # Path to the input Excel file
    output_file = ""  # Path to the output Excel file
    main(input_file, output_file)
