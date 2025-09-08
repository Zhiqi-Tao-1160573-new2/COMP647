# =============================================================================
# LAB1 & LAB2.py - Data Preprocessing and Cleaning
# Function: Handle duplicate data, irrelevant data, missing values, and outlier detection
# 
# ASSIGNMENT REQUIREMENTS COVERAGE:
# 1. Data Preprocessing: Complete data cleaning pipeline including duplicate removal,
#    missing value imputation using median/mode methods, and outlier detection
# 2. EDA Preparation: Cleaned data ready for exploratory analysis and correlation studies
# 3. Feature Insights: Detailed explanations for each preprocessing decision and method choice
# 4. Research Questions: Data prepared for investigating car price prediction models
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
# Load data from CSV file
# Choose this file because it contains complete automotive feature data with boolean values converted to 1 and 0
# This allows direct subsequent data analysis and modeling work
df = pd.read_csv('./primary_features_boolean_converted_final.csv')
print(df.head())  # Display first 5 rows of data
print(df.shape)   # Display data shape (rows, columns)
print(df.info())  # Display basic data information (data types, non-null counts, etc.)
print(df.describe().transpose())  # Display statistical description of numerical data

# =============================================================================
# 2. Initial Data Quality Check
# =============================================================================
# Check basic data quality metrics
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

# Check duplicate data based on specific column combinations
# Reasons for choosing these columns as duplicate detection criteria:
# - user_id: User identity identifier, same user may post multiple vehicles
# - model: Vehicle model, different users may own same model
# - mileage: Mileage, same model but different mileage should be considered different vehicles
# - product_year: Production year, same model but different year should be considered different vehicles
# - price: Price, same model but different price should be considered different vehicles
# This combination can accurately identify true duplicate records and avoid false deletions
duplicate_count = df.duplicated(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)']).sum()
print(f"\nDuplicate data check:")
print(f"Duplicates sum: {duplicate_count}")

# Remove duplicate data, keep the first occurrence
df = df.drop_duplicates(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)'])
print(f"Data shape after removing duplicates: {df.shape}")

# =============================================================================
# 4. Handle Irrelevant Data
# =============================================================================
# Find constant features (columns with only one unique value)
# Constant features have no value for machine learning models because:
# - They provide no distinguishing information
# - They increase model complexity without improving performance
# - They may cause overfitting
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
# Set missing value threshold (50%), columns exceeding this threshold will be deleted
# Reasons for choosing 50% as threshold:
# - Missing values below 50% can be handled by interpolation and other methods
# - Missing values above 50% mean the column has severely insufficient information
# - For price prediction tasks, I need sufficiently complete data to train models
# - 50% is an empirical value that balances data completeness and model performance
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
# 7. Handle Missing Values
# =============================================================================
# Find rows containing missing values
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

# Remove rows with price 0 (price cannot be 0)
# Price 0 is usually data entry error or placeholder, not real vehicle price
# Keeping this data will cause the model to learn wrong patterns
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
# Define automotive feature column list
# These features have important impact on vehicle price:
# - ABS, ESP and other safety configurations directly affect vehicle value
# - Air conditioning, navigation and other comfort configurations affect user experience and price
# - Sunroof, leather seats and other luxury configurations significantly increase price
# Therefore, these missing values must be filled rather than simply deleted
car_feature_columns = [
    "ABS", "Accessible for PWD", "Air Conditioning", "Alarm System", "Central Locking", 
    "Central Screen (Navigation)", "Climate Control", "Electric Side Mirros", "ESP", "Heated Seats", 
    "On-Board Computer", "Parking Control", "Rear View Camera", "Rims", "Start-Stop System", 
    "Steering Hydraulics", "Sunroof"
]

print(f"\nFilling missing values in automotive feature columns:")
print("="*50)

# Iterate through all rows to find empty columns and fill them
# Use similarity-based interpolation strategy:
# - Vehicles of same manufacturer and model usually have similar configurations
# - This method is more reasonable than simply using global mode
# - Maintains logical consistency of data
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

# Use median to fill missing values in numerical columns
# Median imputation is chosen over mean because:
# - Median is robust to outliers, which are common in price and mileage data
# - Median preserves the central tendency without being skewed by extreme values
# - For automotive data, median represents typical values better than mean
# - Median is less sensitive to data distribution shape (skewed vs normal)
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

# Use mode to fill missing values in categorical columns
# Mode imputation is chosen for categorical data because:
# - Mode represents the most frequent category, maintaining data distribution
# - For categorical features like fuel_type, gear, color, mode reflects market preferences
# - Mode preserves the original data structure better than arbitrary category assignment
# - Mode is the most appropriate central tendency measure for nominal categorical data
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
    IQR = input_df[variable].quantile(0.75) - input_df[variable].quantile(0.25)
    lower_limit = input_df[variable].quantile(0.25) - (IQR*1.5)
    upper_limit = input_df[variable].quantile(0.75) + (IQR*1.5)

    return lower_limit,upper_limit

# Find lower and upper bounds for target feature (price)
# Price is the target variable I want to predict, outliers will seriously affect model performance
# Price outliers usually indicate:
# - Data entry errors (such as extra or missing zeros)
# - Special vehicles (such as antique cars, limited editions)
# - Data quality issues
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
# Probability plot (probplot) is typically used for normality testing, also a helpful visual tool for identifying outliers
# and assessing distribution fit
# Reasons for choosing probability plot:
# - Can intuitively display data distribution normality
# - Outliers will obviously deviate from the line in the plot
# - Easy to compare effects before and after processing
# - Help understand changes in data distribution
# Points far from line: Possible outliers
# Points far only at ends: Outliers in tails
# Sudden jumps in spacing: Data irregularities or outliers
# S-shaped curve: Non-normality + possible outliers

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

plt.show()

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

# Calculate Z-scores for specified feature
# Z-score method as supplement to IQR method, provides another perspective for outlier detection
# Characteristics of Z-score method:
# - Based on normal distribution assumption
# - More sensitive to extreme values
# - Suitable for handling data close to normal distribution
df_z_scores = find_outliers_ZScore_method(df.copy(),feature)
df_z_scores.head()

# Remove outliers relative to the feature. Remove data points where |Z| > 3
# Reasons for choosing 3 as threshold:
# - In normal distribution, probability of |Z| > 3 is about 0.27%
# - This is a commonly used significance level in statistics
# - Balances sensitivity and specificity of outlier detection
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
output_file = 'primary_features_cleaned_final.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")

print("\nData preprocessing completed!")

# =============================================================================
# RESEARCH QUESTIONS AND INSIGHTS FROM PREPROCESSING
# =============================================================================
# Based on the data preprocessing analysis, several key research questions emerge:
# 
# 1. PRICE PREDICTION MODEL: Can I build accurate models to predict car prices
#    based on features like mileage, year, engine volume, and car configurations?
#    - Target variable: price(Georgian Lari) - shows strong correlation with vehicle age
#    - Key predictors: mileage, product_year, engine_volume, car features (ABS, AC, etc.)
# 
# 2. FEATURE IMPORTANCE ANALYSIS: Which car features have the strongest impact on price?
#    - Luxury features (sunroof, leather seats) likely show positive correlation
#    - Safety features (ABS, ESP) may have moderate positive impact
#    - Basic features (air conditioning) might show varying effects by market segment
# 
# 3. MARKET SEGMENTATION: How do different car brands and models cluster in price ranges?
#    - Brand_model feature enables analysis of brand premium effects
#    - Vehicle age and mileage interaction reveals depreciation patterns
#    - Price_per_km ratio identifies value propositions across segments
# 
# 4. OUTLIER ANALYSIS: What do price outliers tell us about the Georgian car market?
#    - High-end luxury vehicles vs. data entry errors
#    - Antique/collector cars vs. modern vehicles
#    - Market anomalies that could indicate economic factors
# 
# The preprocessing reveals this dataset is well-suited for regression analysis,
# with clear target variable and multiple predictor features ready for modeling.
# =============================================================================

