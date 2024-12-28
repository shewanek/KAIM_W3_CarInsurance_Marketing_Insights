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

    def univariate_analysis(self):
        """Plot histograms for numerical columns and bar charts for categorical columns."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        # Plot histograms for numerical columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col].dropna(), kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

        # Plot bar charts for categorical columns
        for col in categorical_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.df, y=col, palette='viridis', order=self.df[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.show()

    def bivariate_analysis(self):
        """Explore correlations and associations between key variables."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='TotalPremium', y='TotalClaim', hue='ZipCode', palette='coolwarm')
        plt.title('Total Premium vs Total Claim by ZipCode')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claim')
        plt.show()

        # Correlation matrix
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()

    def data_comparison(self):
        """Compare trends over geography."""
        grouped_data = self.df.groupby('ZipCode')[['TotalPremium', 'TotalClaim']].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=grouped_data, x='ZipCode', y='TotalPremium', label='Average Premium')
        sns.lineplot(data=grouped_data, x='ZipCode', y='TotalClaim', label='Average Claim')
        plt.title('Trends in Premium and Claim by ZipCode')
        plt.xlabel('ZipCode')
        plt.ylabel('Average Value')
        plt.legend()
        plt.show()

    def detect_outliers(self):
        """Use box plots to detect outliers in numerical data."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=self.df, y=col, palette='Set2')
            plt.title(f'Outlier Detection for {col}')
            plt.ylabel(col)
            plt.show()

    def creative_visualizations(self):
        """Produce three creative and beautiful plots."""
        # Example 1: Premium Distribution by Cover Type
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='CoverType', y='TotalPremium', palette='coolwarm')
        plt.title('Premium Distribution by Cover Type')
        plt.xlabel('Cover Type')
        plt.ylabel('Total Premium')
        plt.show()

        # Example 2: Claims vs Premium Heatmap
        pivot_table = self.df.pivot_table(values='TotalClaim', index='AutoMake', columns='CoverType', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Claims vs Cover Type by Auto Make')
        plt.xlabel('Cover Type')
        plt.ylabel('Auto Make')
        plt.show()

        # Example 3: Vehicle Introduction Date Trend
        self.df['VehicleIntroYear'] = self.df['VehicleIntroDate'].dt.year
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x='VehicleIntroYear', y='TotalPremium', label='Total Premium')
        sns.lineplot(data=self.df, x='VehicleIntroYear', y='TotalClaim', label='Total Claim')
        plt.title('Trends Over Vehicle Introduction Years')
        plt.xlabel('Vehicle Introduction Year')
        plt.ylabel('Value')
        plt.legend()
        plt.show()




