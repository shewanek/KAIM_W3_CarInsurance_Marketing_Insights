import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
class InsuranceAnalysis:
    def __init__(self, df):
        """Initialize the analysis class"""
        self.df = df  # Create copy to avoid modifying original
        plt.style.use('seaborn-v0_8')  # Set a better style for visualizations

    def load_data(self):
        """Load the dataset and return a copy"""
        return self.df

    def descriptive_statistics(self):
        """Calculate descriptive statistics for numerical features."""
        return self.df.describe().T

    def check_missing_values(self):
        """Check for missing values in the dataset."""
        # Calculate all metrics in one pass for efficiency
        missing_stats = pd.DataFrame({
            'Missing Values': self.df.isnull().sum(),
            '% of Total Values': 100 * self.df.isnull().sum() / len(self.df),
            'Data type': self.df.dtypes
        })
        return missing_stats.sort_values('% of Total Values', ascending=False).round(2)

    def drop_high_missing_columns(self, threshold=50):
        """
        Drop columns with missing values above the specified threshold.

        Args:
            threshold (float): percentage threshold for dropping columns (default 50%)

        Returns:
            pd.DataFrame: DataFrame with high-missing columns dropped

        Raises:
            ValueError: If threshold is not between 0 and 100
        """
        if not 0 <= threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")

        # Use NumPy for faster computation of missing percentages
        missing_pct = (self.df.isnull().sum(axis=0) / len(self.df)) * 100
        columns_to_drop = missing_pct[missing_pct > threshold].index

        # Drop columns
        if len(columns_to_drop) > 0:
            self.df = self.df.drop(columns=columns_to_drop)
            print(f"Dropped {len(columns_to_drop)} columns: {list(columns_to_drop)}")
        else:
            print("No columns exceeded the missing threshold")

        return self.df



    def date_conversion(self):
        """Convert date columns to datetime format."""
        try:
            self.df['VehicleIntroDate'] = pd.to_datetime(self.df['VehicleIntroDate'], errors='coerce').dt.to_period('M')
            self.df['RegistrationYear'] = pd.to_numeric(self.df['RegistrationYear'], errors='coerce').astype('Int64')
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            return self.df
        except Exception as e:
            print(f"Error converting dates: {str(e)}")
            raise

    