import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#Load Data from a CSV
df = pd.read_csv('D:/Course/647/assement/assessment/primary_features.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().transpose())

#1. Handle Duplicates
df.columns = df.columns.str.strip()
duplicate_count = df.duplicated(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)']).sum()
print(f"Duplicates sum: {duplicate_count}")

df = df.drop_duplicates(subset=['user_id', 'model','mileage','product_year','price(Georgian Lari)'])

#2. Handle Irrelevant Data
constant_features = [col for col in df.columns if df[col].nunique()==1]
print("Constant features: ",constant_features)
df = df.drop(columns=constant_features)

df = df.drop(columns=['user_status'])

#Columns with mostly missing values (e.g., more than x% missing)
#For there are a lot of columns not necessary but relevant to the price, so I set 50% as the drop threshold.
threshold = 50
print(f"Total records {df.shape[0]}")
print("*"*50)
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_ratio = (missing_count/df.shape[0])*100
    if missing_ratio>threshold:
        print(f"Column: {col} has {missing_count} missing values ({missing_ratio:.2f}%)")
        print('*'*50)

#Remove columns with more than x% missing values.
columns_to_drop = [col for col in df.columns if (df[col].isnull().sum()/df.shape[0])*100>threshold]
df_low_missing_data = df.drop(columns=columns_to_drop)

#3. Handle Missing Values
df_missing_data = df[df.isnull().any(axis=1)]
print(df_missing_data.shape)
print(df_missing_data.tail())

#Identify numerical and categorical columns.
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object','category']).columns
print("Numerical columns: ",numerical_columns)
print("Categorical columns: ",categorical_columns)

#Get the list of columns with missing values only for numerical columns.
missing_numerical_columns = df[numerical_columns].isnull().any()
missing_numerical_columns = missing_numerical_columns[missing_numerical_columns].index
print("Numerical columns with missing values: ",missing_numerical_columns.tolist())

#drop the list of rows with 0 only for car price.
#The only necessary column is final price.
price_zero_rows = df[df['price(Georgian Lari)'] == 0]
print("price_zero_rows: ",price_zero_rows.shape)
df = df.drop(index=price_zero_rows.index)

#fulfill the car feature columns
car_feature_columns = [
    "ABS", "Accessible for PWD", "Air Conditioning", "Alarm System", "Central Locking", 
"Central Screen (Navigation)", "Climate Control", "Electric Side Mirros", "ESP", "Heated Seats", 
"On-Board Computer", "Parking Control", "Rear View Camera", "Rims", "Start-Stop System", 
"Steering Hydraulics", "Sunroof"
]

#Iterate all rows to find empty columns
for col in car_feature_columns:
    count = 0 
    for idx, row in df[df[col].isnull()].iterrows():
        # get the data of similar manufacture and model
        similar = df[
            (df["manufacture"] == row["manufacture"]) &
            (df["model"] == row["model"]) &
            (~df[col].isnull())
        ]
        
        # fulfill the empty columns by the data of the same model
        if not similar.empty:
            mode_value = similar[col].mode().iloc[0]
            df.at[idx, col] = mode_value
            print(f'fulfill empty columns {col} of {row["manufacture"]} {row["model"]} with {mode_value}')

        #The iteration takes a long time so I set a limit to it.
        count+=1
        if count>50:
            break


# Interquartile Range: Remove points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
def find_outliers_IQR_method(input_df,variable):
    IQR=input_df[variable].quantile(0.75)-input_df[variable].quantile(0.25)
    lower_limit=input_df[variable].quantile(0.25)-(IQR*1.5)
    upper_limit=input_df[variable].quantile(0.75)+(IQR*1.5)

    return lower_limit,upper_limit

# Find lower and upper limit for target
feature = 'price(Georgian Lari)'
lower,upper = find_outliers_IQR_method(df,feature)
print('lower_limit:',lower," upper_limit:",upper)

#Remove outliers w.r.t the Feature
df_cleaned = df[(df[feature]>lower)&(df[feature]<upper)]
print(f'Cleand dataset : {df_cleaned.shape}')
print(f'Outliers count : {len(df)-len(df_cleaned)}')

#Probability plots before and after handling outliers
# A probability ploy(probplot)-typically used in normality testing, is also a helpful visual tool for indentifying outliers
# and assessing distribution fit.

# Points far from the line        Possible outliers
# Points far at the ends only     Outliers in tails
# Sudden jumps in spacing         Data irregularities or outliers
# S-shape curve                   Non-normality + possible outliers
sns.set_style('whitegrid')
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
stats.probplot(df[feature], plot=plt)

plt.subplot(1,2,2)
stats.probplot(df_cleaned[feature], plot=plt)

plt.show()


def find_outliers_ZScore_method(input_df,variable):
    df_z_scores = input_df.copy()

    # Calculate Z-scores for the pecified variable droping any rows having NaN values
    z_scores = np.abs(stats.zscore(input_df[variable].dropna()))

    # Add Z-scores as a new column
    df_z_scores[variable+'_Z']=z_scores
    return df_z_scores

# Calculate Z-scores for the specified feature
df_z_scores = find_outliers_ZScore_method(df.copy(),feature)
df_z_scores.head()

#Remove outliers w.r.t the Feature. Remove data points where |Z| >3.
df_z_score_cleand = df_z_scores[df_z_scores[feature+'_Z']<3]
print(f'Cleand datase : {df_z_score_cleand.shape}')
print(f'Outliers count : {len(df_z_scores)-len(df_z_score_cleand)}')

sns.set_style('whitegrid')
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
stats.probplot(df[feature], plot=plt)

plt.subplot(1,2,2)
stats.probplot(df_z_score_cleand[feature], plot=plt)

plt.show()

