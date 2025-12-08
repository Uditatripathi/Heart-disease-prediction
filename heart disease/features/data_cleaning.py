"""
Data Cleaning and Preprocessing Module
Handles missing values, outliers, duplicates, and data quality checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Comprehensive data cleaning and preprocessing class"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = {}
        
    def detect_duplicates(self):
        """Detect and report duplicate rows"""
        duplicates = self.df.duplicated().sum()
        self.cleaning_report['duplicates'] = {
            'count': duplicates,
            'percentage': (duplicates / len(self.df)) * 100
        }
        return duplicates
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        self.cleaning_report['duplicates_removed'] = before - after
        return self.df
    
    def detect_missing_values(self):
        """Detect and report missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        self.cleaning_report['missing_values'] = missing_df.to_dict('records')
        return missing_df
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values based on strategy"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        self.cleaning_report['missing_strategy'] = strategy
        return self.df
    
    def detect_outliers_zscore(self, threshold=3):
        """Detect outliers using Z-score method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col]))
            outliers = np.where(z_scores > threshold)[0]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'indices': outliers.tolist()
            }
        
        self.cleaning_report['outliers_zscore'] = outlier_info
        return outlier_info
    
    def detect_outliers_iqr(self):
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        self.cleaning_report['outliers_iqr'] = outlier_info
        return outlier_info
    
    def remove_outliers_zscore(self, threshold=3):
        """Remove outliers using Z-score method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        before = len(self.df)
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col]))
            self.df = self.df[z_scores < threshold]
        
        after = len(self.df)
        self.cleaning_report['outliers_removed'] = before - after
        return self.df
    
    def get_correlation_matrix(self):
        """Generate correlation matrix"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        return correlation_matrix
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """Plot correlation heatmap"""
        correlation_matrix = self.get_correlation_matrix()
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return plt
    
    def get_cleaning_summary(self):
        """Get comprehensive cleaning summary"""
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'cleaning_report': self.cleaning_report
        }
        return summary
    
    def get_cleaned_data(self):
        """Return cleaned dataframe"""
        return self.df


def load_sample_data():
    """Generate sample heart disease dataset if no data file exists"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
        'restecg': np.random.randint(0, 3, n_samples),  # resting ECG
        'thalach': np.random.randint(71, 202, n_samples),  # max heart rate
        'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),  # slope of peak exercise
        'ca': np.random.randint(0, 4, n_samples),  # number of major vessels
        'thal': np.random.randint(0, 3, n_samples),  # thalassemia
        'target': np.random.randint(0, 2, n_samples)  # target variable
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    df.loc[df['age'] > 60, 'target'] = np.random.choice([0, 1], p=[0.3, 0.7], size=len(df[df['age'] > 60]))
    df.loc[df['chol'] > 250, 'target'] = np.random.choice([0, 1], p=[0.4, 0.6], size=len(df[df['chol'] > 250]))
    df.loc[df['trestbps'] > 140, 'target'] = np.random.choice([0, 1], p=[0.4, 0.6], size=len(df[df['trestbps'] > 140]))
    
    return df

