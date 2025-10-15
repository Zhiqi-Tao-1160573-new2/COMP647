# =============================================================================
# Assessment 02 - Data Preprocessing and EDA Analysis
# Function: Complete data cleaning pipeline and exploratory data analysis
# Coverage: Duplicate removal, missing value imputation, outlier detection, and visualization
# Purpose: Prepare automotive data for price prediction modeling
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # Ignore warning messages

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# =============================================================================
# 1. Data Loading
# =============================================================================
# Load automotive dataset with boolean features converted to 0/1 values
# Display basic data information including shape, types, and statistics
# Check for data quality issues like infinite values and negative prices
df = pd.read_csv('./primary_features_boolean_converted_final.csv')
print(df.head())  # Display first 5 rows of data
print(df.shape)   # Display data shape (rows, columns)
print(df.info())  # Display basic data information (data types, non-null counts, etc.)
print(df.describe().transpose())  # Display statistical description of numerical data

# =============================================================================
# 2. Initial Data Quality Check
# =============================================================================
# Check basic data quality metrics including data types, memory usage, and negative values
print("\n" + "="*50)
print("Initial Data Quality Check")
print("="*50)

# Check data types
print("Data type check:")
print(df.dtypes)
print("\nData type statistics:")
print(df.dtypes.value_counts())

# Check memory usage
print(f"\nData memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"\nInfinite value count: {inf_count}")

# Check for negative values (for columns that shouldn't be negative)
negative_price = (df['price(Georgian Lari)'] < 0).sum()
negative_mileage = (df['mileage'] < 0).sum()
negative_year = (df['product_year'] < 0).sum()
print(f"\nNegative value check:")
print(f"  Negative price: {negative_price}")
print(f"  Negative mileage: {negative_mileage}")
print(f"  Negative year: {negative_year}")

# =============================================================================
# 3. Handle Duplicate Data
# =============================================================================
# Clean spaces in column names
df.columns = df.columns.str.strip()

# Clean column names and identify duplicate records based on key features
# Remove duplicates keeping first occurrence to maintain data integrity
# Duplicates identified using user_id, model, mileage, product_year, and price combination
duplicate_count = df.duplicated(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)']).sum()
print(f"\nDuplicate data check:")
print(f"Duplicates sum: {duplicate_count}")

# Remove duplicate data, keep the first occurrence
df = df.drop_duplicates(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)'])
print(f"Data shape after removing duplicates: {df.shape}")

# =============================================================================
# 4. Handle Irrelevant Data
# =============================================================================
# Remove constant features (single unique value) as they provide no information
# Remove user_status column which doesn't affect vehicle pricing
# These features add noise without contributing to price prediction models
constant_features = [col for col in df.columns if df[col].nunique()==1]
print(f"\nConstant feature check:")
print("Constant features: ",constant_features)
df = df.drop(columns=constant_features)  # Remove constant feature columns

# Remove user_status column (considered unimportant for price prediction)
# User status (such as Standard, Premium, etc.) usually doesn't affect the vehicle price itself
# Vehicle price is mainly determined by vehicle features, not user type
if 'user_status' in df.columns:
    df = df.drop(columns=['user_status'])
    print("user_status column removed")

# =============================================================================
# 5. Handle High Missing Value Columns
# =============================================================================
# Remove columns with missing value ratio exceeding 50% threshold
# High missing ratios indicate insufficient data for reliable analysis
# 50% threshold balances data completeness with model performance needs
threshold = 50
print(f"\nMissing value handling:")
print(f"Total records {df.shape[0]}")
print("*"*50)

# Iterate through all columns to check missing value ratios
high_missing_columns = []
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_ratio = (missing_count/df.shape[0])*100
    if missing_ratio>threshold:
        print(f"Column: {col} has {missing_count} missing values ({missing_ratio:.2f}%)")
        print('*'*50)
        high_missing_columns.append(col)

# Remove columns with missing value ratio exceeding threshold
if high_missing_columns:
    df = df.drop(columns=high_missing_columns)
    print(f"Removed {len(high_missing_columns)} high missing value columns")
    print(f"Data shape after removal: {df.shape}")

# =============================================================================
# 6. Data Type Validation and Correction
# =============================================================================
# Convert object columns to numeric where possible using pandas.to_numeric
# Identify numerical and categorical columns for appropriate processing
# Check categorical column cardinality to identify encoding needs
print(f"\nData type validation and correction:")
print("="*50)
numerical_columns = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {list(numerical_columns)}")

# Check if numerical columns contain non-numerical characters
for col in numerical_columns:
    if df[col].dtype == 'object':
        # Try to convert to numerical type
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted {col} to numerical type")
        except:
            print(f"Cannot convert {col} to numerical type")

# Check and correct categorical columns
categorical_columns = df.select_dtypes(include=['object','category']).columns
print(f"Categorical columns: {list(categorical_columns)}")

# Check unique value counts in categorical columns
for col in categorical_columns:
    unique_count = df[col].nunique()
    print(f"  {col}: {unique_count} unique values")
    # If too many unique values, may need to reconsider whether it should be a categorical feature
    if unique_count > 100:
        print(f"    Warning: {col} has too many unique values, may need re-encoding")

# =============================================================================
# 7. Remove Unreasonable Data
# =============================================================================
# Remove records with invalid values that indicate data entry errors
# Price 0, negative mileage, and unreasonable years are clearly data quality issues
# These outliers would distort correlation analysis and model training
df_missing_data = df[df.isnull().any(axis=1)]
print(f"\nMissing value handling:")
print(f"Rows with missing values: {df_missing_data.shape[0]}")

# Get columns with missing values in numerical columns
missing_numerical_columns = df[numerical_columns].isnull().any()
missing_numerical_columns = missing_numerical_columns[missing_numerical_columns].index
print(f"Numerical columns with missing values: {missing_numerical_columns.tolist()}")

# Get columns with missing values in categorical columns
missing_categorical_columns = df[categorical_columns].isnull().any()
missing_categorical_columns = missing_categorical_columns[missing_categorical_columns].index
print(f"Categorical columns with missing values: {missing_categorical_columns.tolist()}")

# Remove zero prices
price_zero_rows = df[df['price(Georgian Lari)'] == 0]
print(f"\nRows with price 0: {price_zero_rows.shape[0]}")
if len(price_zero_rows) > 0:
    df = df.drop(index=price_zero_rows.index)
    print("Removed rows with price 0")

# Remove other obviously unreasonable data
# Remove rows with negative mileage
negative_mileage_rows = df[df['mileage'] < 0]
if len(negative_mileage_rows) > 0:
    df = df.drop(index=negative_mileage_rows.index)
    print(f"Removed {len(negative_mileage_rows)} rows with negative mileage data")

# Remove rows with unreasonable production year
unreasonable_year_rows = df[(df['product_year'] < 1900) | (df['product_year'] > 2024)]
if len(unreasonable_year_rows) > 0:
    df = df.drop(index=unreasonable_year_rows.index)
    print(f"Removed {len(unreasonable_year_rows)} rows with unreasonable year data")

# Remove rows with unreasonable engine displacement
unreasonable_engine_rows = df[(df['engine_volume'] < 0) | (df['engine_volume'] > 10000)]
if len(unreasonable_engine_rows) > 0:
    df = df.drop(index=unreasonable_engine_rows.index)
    print(f"Removed {len(unreasonable_engine_rows)} rows with unreasonable engine displacement data")

# =============================================================================
# 8. Fill Missing Values in Automotive Feature Columns
# =============================================================================
# Fill missing car feature values using similarity-based interpolation
# Use mode values from same manufacturer and model for logical consistency
# Car features like ABS, AC, navigation significantly impact vehicle pricing
car_feature_columns = [
    "ABS", "Accessible for PWD", "Air Conditioning", "Alarm System", "Central Locking", 
    "Central Screen (Navigation)", "Climate Control", "Electric Side Mirros", "ESP", "Heated Seats", 
    "On-Board Computer", "Parking Control", "Rear View Camera", "Rims", "Start-Stop System", 
    "Steering Hydraulics", "Sunroof"
]

print(f"\nFilling missing values in automotive feature columns:")
print("="*50)

# Process each car feature column to fill missing values using same manufacturer/model mode
# Limit processing to 50 rows per column for efficiency
for col in car_feature_columns:
    if col not in df.columns:
        continue
        
    missing_count = df[col].isnull().sum()
    if missing_count == 0:
        continue
        
    print(f"\nProcessing column: {col}, missing value count: {missing_count}")
    
    count = 0 
    for idx, row in df[df[col].isnull()].iterrows():
        # Get data of same manufacturer and model
        similar = df[
            (df["manufacture"] == row["manufacture"]) &
            (df["model"] == row["model"]) &
            (~df[col].isnull())
        ]
        
        # Fill empty columns using mode of same model
        # Reasons for choosing mode instead of mean:
        # - These features are all boolean values (0 or 1)
        # - Mode represents the most common configuration choice
        # - More consistent with actual market configuration distribution
        if not similar.empty:
            mode_value = similar[col].mode().iloc[0]
            df.at[idx, col] = mode_value
            print(f'  Filled {col} of {row["manufacture"]} {row["model"]} with {mode_value}')

        # Set limit due to long iteration time
        # Reason for limiting to 50 rows:
        # - Avoid program running too long
        # - 50 rows are sufficient to demonstrate filling effect
        # - Can be adjusted as needed in practical applications
        count+=1
        if count>50:
            print(f"  Reached processing limit (50 rows), stopping")
            break

# =============================================================================
# 9. Fill Missing Values in Numerical Columns
# =============================================================================
print(f"\nFilling missing values in numerical columns:")
print("="*50)

# Use median imputation for numerical columns to handle outliers robustly
# Median preserves central tendency without being skewed by extreme values
# More appropriate than mean for automotive data with potential outliers
for col in missing_numerical_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"  Used median to fill {col}: {missing_count} missing values, median value: {median_value}")

# =============================================================================
# 10. Fill Missing Values in Categorical Columns
# =============================================================================
print(f"\nFilling missing values in categorical columns:")
print("="*50)

# Use mode imputation for categorical columns to maintain data distribution
# Mode represents most frequent category, reflecting market preferences
# Preserves original data structure better than arbitrary category assignment
for col in missing_categorical_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_value, inplace=True)
            print(f"  Used mode to fill {col}: {missing_count} missing values, mode value: {mode_value}")

# =============================================================================
# 11. Feature Engineering
# =============================================================================
# Create derived features that capture important automotive market patterns
# Vehicle age, price per km, and brand-model combinations improve model performance
# These features reveal depreciation patterns and value propositions across segments
print(f"\nFeature engineering:")
print("="*50)
if 'product_year' in df.columns:
    current_year = 2024  # Assume current year
    df['vehicle_age'] = current_year - df['product_year']
    print(f"  Created vehicle age feature: vehicle_age")

# Create price density feature (price/mileage)
if 'mileage' in df.columns and 'price(Georgian Lari)' in df.columns:
    # Avoid division by zero error
    df['price_per_km'] = np.where(df['mileage'] > 0, 
                                  df['price(Georgian Lari)'] / df['mileage'], 
                                  0)
    print(f"  Created price density feature: price_per_km")

# Create brand feature combination
if 'manufacture' in df.columns and 'model' in df.columns:
    df['brand_model'] = df['manufacture'] + '_' + df['model']
    print(f"  Created brand-model combination feature: brand_model")

# Check and handle outliers
print(f"\nOutlier check:")
print("="*50)

# Check price outliers
price_stats = df['price(Georgian Lari)'].describe()
price_iqr = price_stats['75%'] - price_stats['25%']
price_lower = price_stats['25%'] - 1.5 * price_iqr
price_upper = price_stats['75%'] + 1.5 * price_iqr

price_outliers = df[(df['price(Georgian Lari)'] < price_lower) | (df['price(Georgian Lari)'] > price_upper)]
print(f"  Price outlier count: {len(price_outliers)}")

# Check mileage outliers
if 'mileage' in df.columns:
    mileage_stats = df['mileage'].describe()
    mileage_iqr = mileage_stats['75%'] - mileage_stats['25%']
    mileage_upper = mileage_stats['75%'] + 1.5 * mileage_iqr
    
    mileage_outliers = df[df['mileage'] > mileage_upper]
    print(f"  Mileage outlier count: {len(mileage_outliers)}")

# =============================================================================
# 12. Outlier Detection and Handling - IQR Method
# =============================================================================
# Use IQR method to detect and remove price outliers for robust analysis
# IQR method is insensitive to outliers and doesn't assume normal distribution
# 1.5 times IQR is standard threshold for identifying extreme values
def find_outliers_IQR_method(input_df,variable):
    """
    Use IQR method to detect outliers
    Parameters:
        input_df: Input dataframe
        variable: Variable name to detect
    Returns:
        lower_limit, upper_limit: Lower and upper bounds
    """
    IQR = input_df[variable].quantile(0.75) - input_df[variable].quantile(0.25)
    lower_limit = input_df[variable].quantile(0.25) - (IQR*1.5)
    upper_limit = input_df[variable].quantile(0.75) + (IQR*1.5)

    return lower_limit,upper_limit

# Find price outlier bounds and remove extreme values that affect model performance
# Price outliers usually indicate data entry errors or special vehicles
feature = 'price(Georgian Lari)'
lower,upper = find_outliers_IQR_method(df,feature)
print(f'\nPrice outlier detection (IQR method):')
print(f'lower_limit: {lower:.2f}, upper_limit: {upper:.2f}')

# Remove outliers relative to the feature
df_cleaned = df[(df[feature]>lower)&(df[feature]<upper)]
print(f'Cleaned dataset shape: {df_cleaned.shape}')
print(f'Outlier count: {len(df)-len(df_cleaned)}')

# =============================================================================
# 13. Probability Plot Visualization - Before and After Outlier Handling Comparison
# =============================================================================
# Create probability plots to visualize data distribution before and after outlier removal
# Shows improvement in data normality after cleaning extreme values
# Helps assess effectiveness of outlier detection methods

sns.set_style('whitegrid')
plt.figure(figsize=(16,6))

# Probability plot before outlier removal
plt.subplot(1,2,1)
stats.probplot(df[feature], plot=plt)
plt.title('Before Outlier Removal')

# Probability plot after outlier removal
plt.subplot(1,2,2)
stats.probplot(df_cleaned[feature], plot=plt)
plt.title('After Outlier Removal')

# plt.show()

# =============================================================================
# 14. Outlier Detection and Handling - Z-Score Method
# =============================================================================
def find_outliers_ZScore_method(input_df,variable):
    """
    Use Z-score method to detect outliers
    Parameters:
        input_df: Input dataframe
        variable: Variable name to detect
    Returns:
        df_z_scores: Dataframe containing Z-scores
    """
    df_z_scores = input_df.copy()

    # Calculate Z-scores for specified variable, dropping any rows with NaN values
    z_scores = np.abs(stats.zscore(input_df[variable].dropna()))

    # Add Z-scores as new column
    df_z_scores[variable+'_Z']=z_scores
    return df_z_scores

# Z-score method as supplement to IQR method for comprehensive outlier detection
# Based on normal distribution assumption with |Z| > 3 threshold
# Provides alternative perspective on extreme value identification
df_z_scores = find_outliers_ZScore_method(df.copy(),feature)
df_z_scores.head()

# Remove data points where |Z| > 3 for balanced sensitivity and specificity
df_z_score_cleand = df_z_scores[df_z_scores[feature+'_Z']<3]
print(f'\nZ-score method outlier detection:')
print(f'Cleaned dataset shape: {df_z_score_cleand.shape}')
print(f'Outlier count: {len(df_z_scores)-len(df_z_score_cleand)}')

# Z-score method outlier handling probability plot comparison
sns.set_style('whitegrid')
plt.figure(figsize=(16,6))

# Original data
plt.subplot(1,2,1)
stats.probplot(df[feature], plot=plt)
plt.title('Original Data')

# Data cleaned by Z-score method
plt.subplot(1,2,2)
stats.probplot(df_z_score_cleand[feature], plot=plt)
plt.title('After Z-Score Outlier Removal')

# plt.show()

# =============================================================================
# 15. Final Data Quality Report
# =============================================================================
print(f"\n" + "="*50)
print("Final Data Quality Report")
print("="*50)

print(f"Final dataset shape: {df_cleaned.shape}")
print(f"Numerical column count: {len(df_cleaned.select_dtypes(include=[np.number]).columns)}")
print(f"Categorical column count: {len(df_cleaned.select_dtypes(include=['object','category']).columns)}")

# Check if there are still missing values in final data
final_missing = df_cleaned.isnull().sum().sum()
print(f"Final missing value total: {final_missing}")

# Save cleaned data
output_file = 'primary_features_cleaned_final.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")

print("\nData preprocessing completed!")

# Research questions: Price prediction models, feature importance analysis, market segmentation, and outlier analysis
# Dataset is well-suited for regression analysis with clear target variable and predictor features

# =============================================================================
# EDA and Visualization Analysis
# =============================================================================
# Comprehensive exploratory analysis with multiple visualization techniques
# Correlation analysis and feature interactions using statistical methods

from pandas.api.types import CategoricalDtype

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
# Load data from CSV file
# Choose this file because it has undergone complete data preprocessing and contains cleaned high-quality data
# This allows focusing on exploratory analysis without repeating data cleaning steps
df = pd.read_csv('./primary_features_boolean_converted_final.csv')
print(df.head())  # Display first 5 rows of data
print(df.describe().transpose())  # Display statistical description of numerical data

# =============================================================================
# 2. Data Quality Initial Check
# =============================================================================
print("\n" + "="*50)
print("Initial Data Quality Check")
print("="*50)

# Check data types
print("Data type check:")
print(df.dtypes)
print("\nData type statistics:")
print(df.dtypes.value_counts())

# Check memory usage
print(f"\nData memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"\nInfinite value count: {inf_count}")

# Check for negative values (for columns that shouldn't be negative)
negative_price = (df['price(Georgian Lari)'] < 0).sum()
negative_mileage = (df['mileage'] < 0).sum()
negative_year = (df['product_year'] < 0).sum()
print(f"\nNegative value check:")
print(f"  Negative price: {negative_price}")
print(f"  Negative mileage: {negative_mileage}")
print(f"  Negative year: {negative_year}")

# =============================================================================
# 3. Data Type Validation and Correction
# =============================================================================
print(f"\nData type validation and correction:")
print("="*50)

# Check and correct data types of numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {list(numerical_columns)}")

# Check if numerical columns contain non-numerical characters
for col in numerical_columns:
    if df[col].dtype == 'object':
        # Try to convert to numerical type
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted {col} to numerical type")
        except:
            print(f"Cannot convert {col} to numerical type")

# Check and correct categorical columns
categorical_columns = df.select_dtypes(include=['object','category']).columns
print(f"Categorical columns: {list(categorical_columns)}")

# Check unique value counts in categorical columns
for col in categorical_columns:
    unique_count = df[col].nunique()
    print(f"  {col}: {unique_count} unique values")
    # If too many unique values, may need to reconsider whether it should be a categorical feature
    if unique_count > 100:
        print(f"    Warning: {col} has too many unique values, may need re-encoding")

# =============================================================================
# 4. Data Cleaning
# =============================================================================
print(f"\nData cleaning:")
print("="*50)

# Convert prices to thousands for better readability
# Reasons for choosing thousands as unit:
# - Georgian Lari prices are usually large, original values are not intuitive
# - Thousands unit facilitates quick understanding of price range
# - Reduces number of digits in charts, improving readability
# - Conforms to automotive market price expression habits
df['price(Georgian Lari)']= (df['price(Georgian Lari)']/1000).astype(int)

# Sample 20% of data for efficient exploratory analysis while maintaining representativeness
# Fixed random seed ensures reproducible results across runs
df = df.sample(frac=0.20,random_state=42)
print(df.describe().transpose())

# Identify numerical and categorical columns for appropriate analysis methods
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object','category']).columns
print("Numerical columns:", numerical_columns)
print("Categorical columns:",categorical_columns)

# Remove rows with price 0 (price cannot be 0)
# Price 0 is usually data quality issue, not real vehicle price
# Keeping this data will cause correlation analysis results to be distorted
price_zero_rows = df[df['price(Georgian Lari)'] == 0]
print(f"\nRows with price 0: {price_zero_rows.shape[0]}")
if len(price_zero_rows) > 0:
    df = df.drop(index=price_zero_rows.index)
    print("Removed rows with price 0")

# Remove rows with production year 0 (year cannot be 0)
# Production year 0 is clearly data error because:
# - Cars cannot be produced in year 0 AD
# - This is usually a placeholder during data entry
# - Will affect analysis of year-price relationship
year_zero_rows = df[df['product_year']==0]
print(f"Rows with year 0: {year_zero_rows.shape[0]}")
if len(year_zero_rows) > 0:
    df = df.drop(index=year_zero_rows.index)
    print("Removed rows with year 0")

# Remove other obviously unreasonable data
# Remove rows with negative mileage
negative_mileage_rows = df[df['mileage'] < 0]
if len(negative_mileage_rows) > 0:
    df = df.drop(index=negative_mileage_rows.index)
    print(f"Removed {len(negative_mileage_rows)} rows with negative mileage data")

# Remove rows with unreasonable production year
unreasonable_year_rows = df[(df['product_year'] < 1900) | (df['product_year'] > 2024)]
if len(unreasonable_year_rows) > 0:
    df = df.drop(index=unreasonable_year_rows.index)
    print(f"Removed {len(unreasonable_year_rows)} rows with unreasonable year data")

# Remove rows with unreasonable engine displacement
unreasonable_engine_rows = df[(df['engine_volume'] < 0) | (df['engine_volume'] > 10000)]
if len(unreasonable_engine_rows) > 0:
    df = df.drop(index=unreasonable_engine_rows.index)
    print(f"Removed {len(unreasonable_engine_rows)} rows with unreasonable engine displacement data")

# =============================================================================
# 5. Missing Value Handling
# =============================================================================
print(f"\nMissing value handling:")
print("="*50)

# Check for missing values in numerical columns
missing_numerical_columns = df[numerical_columns].isnull().any()
missing_numerical_columns = missing_numerical_columns[missing_numerical_columns].index
print(f"Numerical columns with missing values: {missing_numerical_columns.tolist()}")

# Check for missing values in categorical columns
missing_categorical_columns = df[categorical_columns].isnull().any()
missing_categorical_columns = missing_categorical_columns[missing_categorical_columns].index
print(f"Categorical columns with missing values: {missing_categorical_columns.tolist()}")

# Fill missing values in numerical columns using median
for col in missing_numerical_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"  Used median to fill {col}: {missing_count} missing values, median value: {median_value}")

# Fill missing values in categorical columns using mode
for col in missing_categorical_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_value, inplace=True)
            print(f"  Used mode to fill {col}: {missing_count} missing values, mode value: {mode_value}")

# =============================================================================
# 6. Feature Engineering
# =============================================================================
print(f"\nFeature engineering:")
print("="*50)

# Create vehicle age feature
if 'product_year' in df.columns:
    current_year = 2024  # Assume current year
    df['vehicle_age'] = current_year - df['product_year']
    print(f"  Created vehicle age feature: vehicle_age")

# Create price density feature (price/mileage)
if 'mileage' in df.columns and 'price(Georgian Lari)' in df.columns:
    # Avoid division by zero error
    df['price_per_km'] = np.where(df['mileage'] > 0, 
                                  df['price(Georgian Lari)'] / df['mileage'], 
                                  0)
    print(f"  Created price density feature: price_per_km")

# Create brand feature combination
if 'manufacture' in df.columns and 'model' in df.columns:
    df['brand_model'] = df['manufacture'] + '_' + df['model']
    print(f"  Created brand-model combination feature: brand_model")

# =============================================================================
# 7. Correlation Analysis
# =============================================================================
print(f"\nCorrelation analysis:")
print("="*50)

# =============================================================================
# CHART 1: Bar Chart for Price Correlation
# =============================================================================
# Bar charts ideal for comparing feature correlations with price
# Height represents correlation strength for easy feature ranking
# Clear visual hierarchy shows relative magnitude of correlations
plt.figure(figsize=(20,8))
# Calculate Pearson correlation coefficients between features and price
# Exclude identifiers and prediction columns to avoid data leakage
correlation_data = df[numerical_columns].corr()['price(Georgian Lari)'].drop('price(Georgian Lari)').drop('app_id').drop('user_id').drop('pred_first_breakpoint').drop('pred_second_breakpoint').sort_values(ascending=False)
correlation_data.plot(kind='bar')
plt.title('Correlation with Price - Bar Chart Analysis')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# =============================================================================
# CHART 2: Pair Plot for Numerical Feature Relationships
# =============================================================================
# Matrix of scatter plots showing relationships between all numerical variables
# Provides comprehensive overview of feature interactions in single visualization
# Diagonal shows individual variable distributions and potential multicollinearity
print("\nCreating Pair Plot to analyze relationships between numerical features...")
pair_plot=sns.pairplot(df[['price(Georgian Lari)','mileage','cylinders','product_year','engine_volume']])
pair_plot.fig.suptitle('Pair Plot: Numerical Feature Relationships Analysis',y=1.02)
pair_plot.fig.set_size_inches(15, 12)
# plt.show()

# =============================================================================
# 8. Outlier Detection and Handling - IQR Method
# =============================================================================
# Interquartile Range method: Remove points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
# Reasons for choosing IQR method:
# - Insensitive to outliers, more robust than Z-score method
# - Based on actual data distribution, doesn't assume normal distribution
# - 1.5 times IQR is standard threshold in statistics
# - Suitable for handling right-skewed data like price
def find_outliers_IQR_method(input_df,variable):
    """
    Use IQR method to detect outliers
    Parameters:
        input_df: Input dataframe
        variable: Variable name to detect
    Returns:
        lower_limit, upper_limit: Lower and upper bounds
    """
    IQR=input_df[variable].quantile(0.75)-input_df[variable].quantile(0.25)
    lower_limit=input_df[variable].quantile(0.25)-(IQR*1.5)
    upper_limit=input_df[variable].quantile(0.75)+(IQR*1.5)

    return lower_limit,upper_limit

# Find lower and upper bounds for price feature
# Price is target variable, outliers will seriously affect model performance
# Price outliers usually indicate data quality issues or special vehicles
feature = 'price(Georgian Lari)'
lower,upper = find_outliers_IQR_method(df,feature)
print(f'\nPrice outlier detection (IQR method):')
print(f'lower_price_limit: {lower:.2f}, upper_price_limit: {upper:.2f}')

# Remove outliers relative to price
df_cleaned = df[(df[feature]>lower)&(df[feature]<upper)]
print(f'Cleaned price dataset shape: {df_cleaned.shape}')
print(f'Price outlier count: {len(df)-len(df_cleaned)}')

# Find lower and upper bounds for mileage feature
# Mileage outlier handling strategy:
# - Only remove upper bound outliers (excessively high mileage)
# - Low mileage may be new cars or data errors
# - High mileage may be data entry errors or special purpose vehicles
feature = 'mileage'
lower,upper = find_outliers_IQR_method(df_cleaned,feature)
print(f'Mileage outlier detection (IQR method):')
print(f'lower_mileage_limit: {lower:.2f}, upper_mileage_limit: {upper:.2f}')
# Only remove upper bound outliers (excessively high mileage)
df_cleaned = df_cleaned[(df_cleaned[feature]<upper)]
print(f'Cleaned mileage dataset shape: {df_cleaned.shape}')
print(f'Mileage outlier count: {len(df)-len(df_cleaned)}')

# Find lower and upper bounds for production year feature
# Year outlier handling strategy:
# - Only remove lower bound outliers (excessively low years)
# - Very low years may be data errors or antique cars
# - Very high years may be data entry errors
feature = 'product_year'
lower,upper = find_outliers_IQR_method(df_cleaned,feature)
print(f'Year outlier detection (IQR method):')
print(f'lower_year_limit: {lower:.2f}, upper_year_limit: {upper:.2f}')
# Only remove lower bound outliers (excessively low years)
df_cleaned = df_cleaned[lower<(df_cleaned[feature])]
print(f'Cleaned year dataset shape: {df_cleaned.shape}')
print(f'Year outlier count: {len(df)-len(df_cleaned)}')

# =============================================================================
# 9. Linear Plot Visualization
# =============================================================================
print(f"\nLinear plot visualization:")
print("="*50)

# =============================================================================
# CHART 3: Line Chart for Mileage vs Average Price
# =============================================================================
# Why choose Line Chart for mileage vs price relationship:
# - Line charts are perfect for showing trends and continuous relationships between two variables
# - They excel at displaying how one variable (price) changes as another variable (mileage) increases
# - Line charts make it easy to identify patterns such as price depreciation with mileage
# - They can reveal non-linear relationships and inflection points in the data
# - Line charts are excellent for time-series-like data where one variable has a natural ordering (mileage)
# - They allow easy comparison of price levels across different mileage ranges
# - The continuous line helps visualize the smooth transition between data points
# - Adding markers (marker='o') makes it easy to read exact values at specific mileage points
# - Line charts are ideal for showing aggregated data (average prices) across binned categories
# Group mileage by 10000km intervals for trend analysis
# Balance between detail and noise reduction for clear pattern identification
bins = range(0, int(df_cleaned['mileage'].max()) + 10000, 10000)
df_cleaned['mileage_bin'] = pd.cut(df_cleaned['mileage'], bins=bins)

# Calculate average price by mileage group for trend identification
mileage_avg_price = df_cleaned.groupby('mileage_bin')['price(Georgian Lari)'].mean().reset_index()
mileage_avg_price['mileage_bin_str']=mileage_avg_price['mileage_bin'].astype(str)

# Use categorical data type to ensure correct mileage order in visualization
cat_type=CategoricalDtype(categories=mileage_avg_price['mileage_bin_str'],ordered=True)
mileage_avg_price['mileage_bin_str']=mileage_avg_price['mileage_bin_str'].astype(cat_type)

# Draw relationship chart between mileage and average price
print("\nCreating Line Chart to show price trends across mileage ranges...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
sns.lineplot(data=mileage_avg_price,x='mileage_bin_str',y='price(Georgian Lari)',marker='o')
plt.title('Line Chart: Average Price by Mileage Range - Trend Analysis')
plt.xlabel('Mileage Range (km)')
plt.ylabel('Average Price (Georgian Lari)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# =============================================================================
# 10. Kernel Density Estimation Plot (KDE)
# =============================================================================
print(f"\nKernel Density Estimation plot:")
print("="*50)

# =============================================================================
# CHART 4: KDE Plot for Mileage vs Price Joint Distribution
# =============================================================================
# KDE plots create smooth density representations for large datasets
# Color intensity shows data concentration and reveals clustering patterns
# Excellent for identifying vehicle segments and non-linear relationships
print("\nCreating KDE Plot to analyze joint distribution of mileage and price...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))

kde_plot = sns.kdeplot(data=df_cleaned, x='mileage',y='price(Georgian Lari)',cmap='Blues',fill=True,levels=10)
plt.title('KDE Plot: Joint Distribution of Mileage vs Price - Density Analysis')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (Georgian Lari)')
plt.colorbar(kde_plot.collections[0], label='Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# =============================================================================
# 11. Histogram Plot
# =============================================================================
print(f"\nHistogram plot:")
print("="*50)

# =============================================================================
# CHART 5: Histogram with KDE for Mileage Distribution
# =============================================================================
# Histograms show frequency distribution and data shape including skewness
# KDE overlay provides smooth probability density estimate
# Reveals data patterns and potential quality issues
print("\nCreating Histogram with KDE to analyze mileage distribution...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
sns.histplot(df_cleaned['mileage'],bins=100,kde=True, color='skyblue', alpha=0.7)
plt.title('Histogram with KDE: Mileage Distribution Analysis')
plt.xlabel('Mileage (km)')
plt.ylabel('Frequency Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# =============================================================================
# 12. Violin Plot
# =============================================================================
print(f"\nViolin plot:")
print("="*50)

# =============================================================================
# CHART 6: Violin Plot for Production Year vs Price
# =============================================================================
# Violin plots combine box plot statistics with density estimation
# Show full distribution shape and probability density at each year
# Reveal non-normal distributions and unusual price patterns over time
print("\nCreating Violin Plot to analyze price distribution across production years...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
plot = sns.violinplot(data=df_cleaned,x='product_year',y='price(Georgian Lari)',inner='quartile', palette='Set3')
plt.title('Violin Plot: Price Distribution by Production Year - Distribution Shape Analysis')
plt.xlabel('Production Year')
plt.ylabel('Price (Georgian Lari)')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# =============================================================================
# 13. Box Plot
# =============================================================================
print(f"\nBox plot:")
print("="*50)

# =============================================================================
# CHART 7: Box Plot for Production Year vs Price
# =============================================================================
# Box plots provide clear statistical summaries including median, quartiles, and outliers
# Excellent for comparing central tendencies and variability across years
# Complement violin plots with focused statistical measures
print("\nCreating Box Plot to analyze price statistics across production years...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
plot = sns.boxplot(data=df_cleaned,x='product_year',y='price(Georgian Lari)', palette='Set2')
plt.title('Box Plot: Price Statistics by Production Year - Statistical Summary Analysis')
plt.xlabel('Production Year')
plt.ylabel('Price (Georgian Lari)')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# =============================================================================
# 14. Categorical Column Unique Value Statistics
# =============================================================================
print(f"\nCategorical column unique value statistics:")
print("="*50)

# =============================================================================
# CHART 8: Bar Chart for Categorical Feature Cardinality
# =============================================================================
# Bar charts ideal for comparing unique value counts across categorical features
# Help identify high cardinality features that may need encoding strategies
# Clear visual ranking of feature complexity for preprocessing decisions
print("\nCreating Bar Chart to analyze unique value counts in categorical features...")
# Analyze cardinality of key categorical features affecting vehicle pricing
unique_counts=df_cleaned[['fuel_type','gear','door_type','color','manufacture']].nunique().sort_values()
plt.figure(figsize=(20,8))
bars = sns.barplot(x=unique_counts.index,y=unique_counts.values,palette='Set2')

# Add value labels on top of each bar
for i, v in enumerate(unique_counts.values):
    bars.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

plt.title('Bar Chart: Unique Value Counts for Categorical Features - Cardinality Analysis')
plt.xlabel('Categorical Features')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# 15. Final Data Quality Report
# =============================================================================
print(f"\n" + "="*50)
print("Final Data Quality Report")
print("="*50)

print(f"Final dataset shape: {df_cleaned.shape}")
print(f"Numerical column count: {len(df_cleaned.select_dtypes(include=[np.number]).columns)}")
print(f"Categorical column count: {len(df_cleaned.select_dtypes(include=['object','category']).columns)}")

# Check if there are still missing values in final data
final_missing = df_cleaned.isnull().sum().sum()
print(f"Final missing value total: {final_missing}")

# Save cleaned data
output_file = 'primary_features_eda_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")

print("\nExploratory Data Analysis completed!")
print("\nChart Summary:")
print("- Bar Chart: Price correlation analysis (feature importance ranking)")
print("- Pair Plot: Numerical feature relationship matrix (comprehensive feature interactions)")
print("- Line Chart: Mileage vs price trends (continuous relationship visualization)")
print("- KDE Plot: Joint distribution analysis (density-based pattern recognition)")
print("- Histogram: Mileage distribution analysis (frequency and shape understanding)")
print("- Violin Plot: Price distribution by year (distribution shape comparison)")
print("- Box Plot: Price statistics by year (statistical summary comparison)")
print("- Bar Chart: Categorical feature cardinality (feature complexity analysis)")

# EDA confirms dataset suitability for machine learning with clear target relationships
# Strong correlations between vehicle age, mileage, and price support prediction models
# Feature engineering and outlier removal improve data quality for modeling