# =============================================================================
# LAB3.py - Exploratory Data Analysis (EDA) and Visualization
# Function: Data exploration, correlation analysis, outlier handling, multiple chart visualizations
# =============================================================================

import pandas as pd
from pandas.api.types import CategoricalDtype
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

# Randomly sample 20% of the dataset for exploratory analysis
# Reasons for choosing 20% sample:
# - Reduce computation time, improve analysis efficiency
# - 20% sample is usually sufficient to reflect overall data characteristics
# - Use fixed random seed to ensure reproducible results
# - In cases of large data volume, sampling analysis is more practical
df = df.sample(frac=0.20,random_state=42)
print(df.describe().transpose())

# Identify numerical and categorical columns
# Distinguishing data types is important for subsequent analysis:
# - Numerical columns can perform correlation analysis and statistical tests
# - Categorical columns are suitable for frequency analysis and visualization
# - Different types of data require different analysis methods
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
# Why choose Bar Chart for correlation analysis:
# - Bar charts are ideal for comparing categorical data (feature names) with numerical values (correlation coefficients)
# - The height of each bar directly represents the correlation strength, making it easy to rank features by importance
# - Bar charts provide clear visual hierarchy - longer bars indicate stronger correlations
# - They are excellent for showing the relative magnitude of correlations between multiple features
# - Bar charts are universally understood and accessible to both technical and non-technical audiences
# - They work well with sorted data, allowing us to easily identify the most and least correlated features
# - The horizontal orientation with rotated labels prevents text overlap even with many features
plt.figure(figsize=(20,8))
# Calculate correlation between numerical columns and price, excluding price itself and some irrelevant columns
# Reasons for excluding these columns:
# - app_id, user_id: Identifiers, no logical relationship with price
# - pred_breakpoints: Prediction-related columns, may cause data leakage
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
# Why choose Pair Plot for feature relationship analysis:
# - Pair plots create a matrix of scatter plots showing relationships between all pairs of numerical variables
# - They provide a comprehensive overview of how each feature relates to every other feature in a single visualization
# - The diagonal shows the distribution of each individual variable, helping identify data patterns and skewness
# - Pair plots are excellent for detecting non-linear relationships, clusters, and outliers that might be missed in correlation analysis
# - They allow us to see the "big picture" of feature interactions, which is crucial for understanding the data structure
# - The matrix format makes it easy to compare relationships across different feature combinations
# - They help identify potential multicollinearity issues between features
# - Pair plots are particularly useful for medium-sized datasets where individual scatter plots would be too numerous
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
# Group mileage by every 10000km
# Reasons for choosing 10000km as grouping interval:
# - Too large interval may mask important trends
# - Too small interval may produce noise
# - 10000km is common mileage node for automotive maintenance
# - Facilitates understanding of price changes in different mileage ranges
bins = range(0, int(df_cleaned['mileage'].max()) + 10000, 10000)
df_cleaned['mileage_bin'] = pd.cut(df_cleaned['mileage'], bins=bins)

# Calculate average price by mileage group
# Reasons for using average price instead of median price:
# - Average price reflects overall price level
# - Easy to identify price trends
# - Average price is more sensitive to outliers, helpful for discovering patterns
mileage_avg_price = df_cleaned.groupby('mileage_bin')['price(Georgian Lari)'].mean().reset_index()
mileage_avg_price['mileage_bin_str']=mileage_avg_price['mileage_bin'].astype(str)

# Order mileage
# Use categorical data type to ensure correct mileage order:
# - Avoid wrong order caused by string sorting
# - Ensure logical arrangement of mileage ranges in charts
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
# Why choose KDE Plot for mileage vs price analysis:
# - KDE plots create smooth, continuous representations of data density, unlike discrete scatter plots
# - They are excellent for large datasets where individual points would create visual clutter
# - KDE plots reveal the underlying probability density of the data, showing where most observations are concentrated
# - The color intensity (using cmap='Blues') provides an intuitive way to understand data density
# - They can identify multiple modes or clusters in the data that might indicate different vehicle segments
# - KDE plots are particularly useful for understanding the joint distribution of two continuous variables
# - They help identify areas of high and low data concentration, revealing data patterns
# - The smooth contours make it easier to spot trends and relationships than raw scatter plots
# - KDE plots are excellent for detecting non-linear relationships and data clustering
# - They provide a more sophisticated view of data structure than basic scatter plots
print("\nCreating KDE Plot to analyze joint distribution of mileage and price...")
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))

sns.kdeplot(data=df_cleaned, x='mileage',y='price(Georgian Lari)',cmap='Blues',fill=True,levels=10)
plt.title('KDE Plot: Joint Distribution of Mileage vs Price - Density Analysis')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (Georgian Lari)')
plt.colorbar(label='Density')
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
# Why choose Histogram with KDE for mileage distribution analysis:
# - Histograms provide discrete bins that show the frequency distribution of mileage values
# - They reveal the shape of the data distribution, including skewness, modality, and range
# - Adding KDE curve (kde=True) provides a smooth, continuous estimate of the underlying probability density
# - Histograms are excellent for identifying data patterns such as clustering around certain mileage values
# - They help understand the distribution of mileage across the dataset, revealing common mileage ranges
# - The combination of histogram bars and KDE curve gives both discrete and continuous perspectives
# - Histograms are particularly useful for understanding the spread and central tendency of numerical data
# - They can reveal potential data quality issues such as unusual spikes or gaps in the distribution
# - The binning process (bins=100) helps reduce noise while maintaining sufficient detail
# - Histograms are fundamental tools for understanding the basic structure of numerical variables
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
# Why choose Violin Plot for production year vs price analysis:
# - Violin plots combine the benefits of box plots with density estimation, showing the full distribution shape
# - They reveal the probability density of price at each year, showing where most vehicles are priced
# - Violin plots can display multiple modes or unusual distributions that box plots might miss
# - They provide more information than box plots while maintaining visual clarity
# - The width of each violin represents the relative frequency of observations at that year
# - Violin plots are excellent for comparing distributions across categorical variables (years)
# - They can reveal non-normal distributions, skewness, and bimodality in the price data
# - The inner='quartile' parameter shows the median and quartiles, providing statistical summaries
# - Violin plots are particularly useful for understanding how price distributions change over time
# - They help identify years with unusual price patterns or market conditions
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
# Why choose Box Plot for production year vs price analysis:
# - Box plots provide clear statistical summaries including median, quartiles, and outliers
# - They are excellent for comparing central tendencies and variability across different years
# - Box plots clearly show the spread and skewness of price data at each production year
# - They can identify outliers that might represent special vehicles or data quality issues
# - Box plots are more focused on statistical measures than violin plots, making them easier to interpret
# - They work well with violin plots as complementary visualizations - box plots for statistics, violin plots for shape
# - Box plots are particularly useful for identifying years with unusual price ranges or variability
# - They can reveal trends in price medians and interquartile ranges over time
# - Box plots are excellent for detecting changes in market conditions or vehicle pricing strategies
# - They provide a clean, professional appearance suitable for business presentations and reports
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
# Why choose Bar Chart for categorical feature cardinality analysis:
# - Bar charts are ideal for comparing counts across different categories (feature names)
# - They provide a clear visual representation of how many unique values each categorical feature has
# - Bar charts make it easy to rank features by their cardinality, identifying high and low cardinality features
# - They help identify potential issues such as features with too many unique values (high cardinality)
# - Bar charts are excellent for showing the relative differences between categorical features
# - They can reveal which features might need encoding strategies (e.g., high cardinality features)
# - The horizontal orientation with rotated labels prevents text overlap even with long feature names
# - Bar charts are universally understood and accessible to all audiences
# - They work well with sorted data, making it easy to identify the most and least complex features
# - Bar charts are perfect for displaying discrete count data like unique value counts
print("\nCreating Bar Chart to analyze unique value counts in categorical features...")
# Get unique value counts in categorical columns
# Reasons for choosing these categorical columns:
# - fuel_type: Fuel type affects usage cost and environmental friendliness
# - gear: Transmission type affects driving experience and price
# - door_type: Number of doors affects practicality and price
# - color: Color affects vehicle appearance and market demand
# - manufacture: Manufacturer affects brand value and reliability
# These features are all important factors affecting vehicle price
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