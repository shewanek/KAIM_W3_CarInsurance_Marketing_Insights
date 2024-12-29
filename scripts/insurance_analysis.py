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



    def data_conversion(self):
        """Convert date columns to datetime format."""
        try:
            # Convert VehicleIntroDate with month/year format
            self.df['VehicleIntroDate'] = pd.to_datetime(
                self.df['VehicleIntroDate'],
                format='%m/%Y',
                errors='coerce'
            ).dt.to_period('M')

            # Convert RegistrationYear to numeric
            self.df['RegistrationYear'] = pd.to_numeric(self.df['RegistrationYear'], errors='coerce').astype('Int64')

            # Convert TransactionMonth to datetime
            self.df['TransactionMonth'] = pd.to_datetime(
                self.df['TransactionMonth'],
                errors='coerce'
            )
            # Handle mixed-type columns if necessary
            self.df['CapitalOutstanding'] = pd.to_numeric(self.df['CapitalOutstanding'], errors='coerce')


            return self.df
        except Exception as e:
            print(f"Error converting dates: {str(e)}")
            raise


    def univariate_analysis(self):
        """Plot histograms for numerical columns and bar charts for categorical columns."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        # Plot histograms for numerical columns
        for i in range(0, len(numerical_cols), 3):
            fig, axes = plt.subplots(1, min(3, len(numerical_cols) - i), figsize=(20, 5))
            if len(numerical_cols) - i == 1:
                axes = [axes]
            for j, col in enumerate(numerical_cols[i:i+3]):
                sns.histplot(data=self.df[col].dropna(), kde=True, bins=30, color='blue', ax=axes[j])
                axes[j].set_title(f'Distribution of {col}')
                axes[j].set_xlabel(col)
                axes[j].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()

        # Plot bar charts for categorical columns
        for i in range(0, len(categorical_cols), 3):
            fig, axes = plt.subplots(1, min(3, len(categorical_cols) - i), figsize=(20, 5))
            if len(categorical_cols) - i == 1:
                axes = [axes]
            for j, col in enumerate(categorical_cols[i:i+3]):
                sns.countplot(data=self.df, y=col, palette='viridis',
                            order=self.df[col].value_counts().index, ax=axes[j])
                axes[j].set_title(f'Distribution of {col}')
                axes[j].set_xlabel('Count')
                axes[j].set_ylabel(col)
            plt.tight_layout()
            plt.show()

    def bivariate_analysis(self):
        """Explore correlations and associations between key variables."""
        # Scatter plot of key numeric relationships
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='CalculatedPremiumPerTerm', y='TotalClaims', hue='CoverType', palette='coolwarm')
        plt.title('Premium per Term vs Total Claims by Cover Type')
        plt.xlabel('Calculated Premium Per Term')
        plt.ylabel('Total Claims')
        plt.show()

        # Correlation matrix for key numeric variables
        numeric_cols = ['CalculatedPremiumPerTerm', 'TotalClaims', 'TotalPremium', 'SumInsured', 'CapitalOutstanding',
                         'kilowatts', 'cubiccapacity', 'Cylinders']
        correlation_matrix = self.df[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Key Variables')
        plt.show()

        # Box plot showing claims distribution by vehicle type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='VehicleType', y='TotalClaims', palette='viridis')
        plt.xticks(rotation=45)
        plt.title('Claims Distribution by Vehicle Type')
        plt.show()

        # Premium distribution by cover category
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.df, x='CoverCategory', y='CalculatedPremiumPerTerm')
        plt.xticks(rotation=45)
        plt.title('Premium Distribution by Cover Category')
        plt.show()

