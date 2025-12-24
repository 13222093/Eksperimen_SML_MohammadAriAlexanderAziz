"""
Automated Preprocessing Script for Ethereum Fraud Detection
Author: Mohammad Ari Alexander Aziz
Description: This script automates the preprocessing pipeline for Ethereum fraud detection dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')


class EthereumFraudPreprocessor:
    """
    A class to handle automated preprocessing of Ethereum fraud detection dataset.
    """

    def __init__(self):
        self.scaler = PowerTransformer()
        self.smote = SMOTE(random_state=42)
        self.drop_features = []

    def load_data(self, file_path):
        """
        Load the raw Ethereum transaction dataset.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file

        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def remove_unnecessary_columns(self, df):
        """
        Remove Index and Address columns as they are not needed for modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe without Index and Address columns
        """
        print("Removing unnecessary columns (Index, Address)...")
        # Since we already used index_col=0 in load_data, we just need to drop Address if it exists
        if 'Address' in df.columns:
            df = df.drop('Address', axis=1)

        # The first two columns after index are typically Index and Address in the original data
        # But since we already set index_col=0, we start from column 2
        if 'Index' in df.columns:
            df = df.drop('Index', axis=1)

        print(f"Shape after removing unnecessary columns: {df.shape}")
        return df

    def handle_categorical_features(self, df):
        """
        Remove categorical features with high cardinality.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe without categorical features
        """
        print("Handling categorical features...")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            print(f"Dropping categorical columns: {categorical_cols}")
            df = df.drop(categorical_cols, axis=1)
        else:
            print("No categorical columns found.")

        print(f"Shape after handling categorical features: {df.shape}")
        return df

    def handle_missing_values(self, df):
        """
        Fill missing values with median.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values filled
        """
        print("Handling missing values...")
        missing_before = df.isnull().sum().sum()
        print(f"Missing values before imputation: {missing_before}")

        df = df.fillna(df.median())

        missing_after = df.isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")

        return df

    def remove_zero_variance_features(self, df):
        """
        Remove features with zero variance.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe without zero variance features
        """
        print("Removing zero variance features...")

        # Exclude target column if present
        feature_cols = [col for col in df.columns if col != 'FLAG']
        variances = df[feature_cols].var()
        zero_var_cols = variances[variances == 0].index.tolist()

        if zero_var_cols:
            print(f"Dropping {len(zero_var_cols)} zero variance features: {zero_var_cols}")
            df = df.drop(zero_var_cols, axis=1)
        else:
            print("No zero variance features found.")

        print(f"Shape after removing zero variance features: {df.shape}")
        return df

    def remove_highly_correlated_features(self, df):
        """
        Remove highly correlated features to reduce multicollinearity.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe with reduced multicollinearity
        """
        print("Removing highly correlated features...")

        # Features to drop based on EDA
        drop_features = [
            'total transactions (including tnx to create contract',
            'total ether sent contracts',
            'max val sent to contract',
            ' ERC20 avg val rec',
            ' ERC20 max val rec',
            ' ERC20 min val rec',
            ' ERC20 uniq rec contract addr',
            'max val sent',
            ' ERC20 avg val sent',
            ' ERC20 min val sent',
            ' ERC20 max val sent',
            ' Total ERC20 tnxs',
            'avg value sent to contract',
            'Unique Sent To Addresses',
            'Unique Received From Addresses',
            'total ether received',
            ' ERC20 uniq sent token name',
            'min value received',
            'min val sent',
            ' ERC20 uniq rec addr'
        ]

        # Only drop features that exist in the dataframe
        existing_drop_features = [f for f in drop_features if f in df.columns]

        if existing_drop_features:
            print(f"Dropping {len(existing_drop_features)} highly correlated features")
            df = df.drop(existing_drop_features, axis=1)
        else:
            print("No highly correlated features to drop.")

        print(f"Shape after removing correlated features: {df.shape}")
        return df

    def remove_low_information_features(self, df):
        """
        Remove features with very low information content.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Dataframe without low-information features
        """
        print("Removing low-information features...")

        low_info_features = ['min value sent to contract', ' ERC20 uniq sent addr.1']
        existing_low_info = [f for f in low_info_features if f in df.columns]

        if existing_low_info:
            print(f"Dropping low-information features: {existing_low_info}")
            df = df.drop(existing_low_info, axis=1)
        else:
            print("No low-information features to drop.")

        print(f"Shape after removing low-information features: {df.shape}")
        return df

    def preprocess_pipeline(self, df):
        """
        Execute the complete preprocessing pipeline.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw input dataframe

        Returns:
        --------
        pd.DataFrame
            Fully preprocessed dataframe
        """
        print("\n" + "="*70)
        print("Starting Preprocessing Pipeline")
        print("="*70 + "\n")

        # Step 1: Remove unnecessary columns
        df = self.remove_unnecessary_columns(df)

        # Step 2: Handle categorical features
        df = self.handle_categorical_features(df)

        # Step 3: Handle missing values
        df = self.handle_missing_values(df)

        # Step 4: Remove zero variance features
        df = self.remove_zero_variance_features(df)

        # Step 5: Remove highly correlated features
        df = self.remove_highly_correlated_features(df)

        # Step 6: Remove low-information features
        df = self.remove_low_information_features(df)

        print("\n" + "="*70)
        print("Preprocessing Pipeline Completed")
        print(f"Final dataset shape: {df.shape}")
        print(f"Final features: {list(df.columns)}")
        print("="*70 + "\n")

        return df

    def save_preprocessed_data(self, df, output_path):
        """
        Save preprocessed data to CSV.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed dataframe
        output_path : str
            Path to save the CSV file
        """
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")

    def prepare_for_modeling(self, df, test_size=0.2, apply_smote=True, random_state=42):
        """
        Prepare data for modeling: split, scale, and optionally apply SMOTE.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed dataframe with FLAG column
        test_size : float
            Proportion of test set (default: 0.2)
        apply_smote : bool
            Whether to apply SMOTE for class balancing (default: True)
        random_state : int
            Random state for reproducibility (default: 42)

        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
            If apply_smote is True, returns resampled training data
        """
        from sklearn.model_selection import train_test_split

        print("\n" + "="*70)
        print("Preparing Data for Modeling")
        print("="*70 + "\n")

        # Separate features and target
        X = df.drop('FLAG', axis=1)
        y = df['FLAG']

        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Feature scaling
        print("\nApplying PowerTransformer for feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply SMOTE if requested
        if apply_smote:
            print("\nApplying SMOTE for class balancing...")
            print(f"Before SMOTE - Training samples: {X_train_scaled.shape[0]}")
            print(f"Class distribution:\n{y_train.value_counts()}")

            X_train_scaled, y_train = self.smote.fit_resample(X_train_scaled, y_train)

            print(f"\nAfter SMOTE - Training samples: {X_train_scaled.shape[0]}")
            print(f"Class distribution:\n{pd.Series(y_train).value_counts()}")

        print("\n" + "="*70)
        print("Data Preparation Completed")
        print("="*70 + "\n")

        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler

    def save_scaler(self, output_path='scaler.pkl'):
        """
        Save the fitted scaler to disk.

        Parameters:
        -----------
        output_path : str
            Path to save the scaler pickle file
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {output_path}")

    def load_scaler(self, input_path='scaler.pkl'):
        """
        Load a fitted scaler from disk.

        Parameters:
        -----------
        input_path : str
            Path to the scaler pickle file
        """
        with open(input_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from: {input_path}")


def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = EthereumFraudPreprocessor()

    # Load data
    input_file = 'transaction_dataset.csv'
    df = preprocessor.load_data(input_file)

    # Run preprocessing pipeline
    df_preprocessed = preprocessor.preprocess_pipeline(df)

    # Save preprocessed data
    output_file = 'ethereum_fraud_preprocessing.csv'
    preprocessor.save_preprocessed_data(df_preprocessed, output_file)

    # Prepare for modeling (with SMOTE)
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_for_modeling(
        df_preprocessed,
        test_size=0.2,
        apply_smote=True,
        random_state=42
    )

    # Save scaler
    preprocessor.save_scaler('scaler.pkl')

    print("\n" + "="*70)
    print("Automation Script Completed Successfully!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"1. {output_file} - Preprocessed dataset")
    print(f"2. scaler.pkl - Fitted PowerTransformer scaler")
    print(f"\nReady for modeling!")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Run the main preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler = main()

    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape if isinstance(y_train, np.ndarray) else len(y_train)}")
    print(f"y_test: {y_test.shape if isinstance(y_test, np.ndarray) else len(y_test)}")
