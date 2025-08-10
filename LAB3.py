import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#Load Data from a CSV
df = pd.read_csv('D:/Course/647/assement/assessment/primary_features.csv')
print(df.head())
print(df.describe().transpose())

# Convert prices to thousands for better readability
df['price(Georgian Lari)']= (df['price(Georgian Lari)']/1000).astype(int)

#Randomly sample 20% of the dataset for exploratory analysis.
df = df.sample(frac=0.20,random_state=42)
print(df.describe().transpose())

numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object','category']).columns
print("Numberical columns:", numerical_columns)
print("Categorical columns:",categorical_columns)

#drop the list of rows with 0 only for car price.
#The only necessary column is final price.
price_zero_rows = df[df['price(Georgian Lari)'] == 0]
print("price_zero_rows: ",price_zero_rows.shape)
df = df.drop(index=price_zero_rows.index)

year_zero_rows = df[df['product_year']==0]
print("year_zero_rows: ",year_zero_rows.shape)
df = df.drop(index=year_zero_rows.index)

#Correlation Bar Plot
## bar plot of correlation with 'price(Georgian Lari)'
plt.figure(figsize=(20,8))
df[numerical_columns].corr()['price(Georgian Lari)'].drop('price(Georgian Lari)').drop('app_id').drop('user_id').drop('pred_first_breakpoint').drop('pred_second_breakpoint').sort_values(ascending=False).plot(kind='bar')
# plt.show()
pair_plot=sns.pairplot(df[['price(Georgian Lari)','mileage','cylinders','product_year','engine_volume']])
pair_plot.fig.suptitle('Pair Plot of Numerical Features',y=1.02)
# plt.show()

# Interquartile Range: Remove points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
def find_outliers_IQR_method(input_df,variable):
    IQR=input_df[variable].quantile(0.75)-input_df[variable].quantile(0.25)
    lower_limit=input_df[variable].quantile(0.25)-(IQR*1.5)
    upper_limit=input_df[variable].quantile(0.75)+(IQR*1.5)

    return lower_limit,upper_limit

# Find lower and upper limit for target
feature = 'price(Georgian Lari)'
lower,upper = find_outliers_IQR_method(df,feature)
print('lower_price_limit:',lower," upper_price_limit:",upper)

#Remove outliers w.r.t the Feature
df_cleaned = df[(df[feature]>lower)&(df[feature]<upper)]
print(f'Cleand price dataset : {df_cleaned.shape}')
print(f'Outliers price count : {len(df)-len(df_cleaned)}')

feature = 'mileage'
lower,upper = find_outliers_IQR_method(df_cleaned,feature)
print('lower_mileage_limit:',lower," upper_mileage_limit:",upper)
df_cleaned = df_cleaned[(df_cleaned[feature]<upper)]
print(f'Cleand mileage dataset : {df_cleaned.shape}')
print(f'Outliers mileage count : {len(df)-len(df_cleaned)}')

feature = 'product_year'
lower,upper = find_outliers_IQR_method(df_cleaned,feature)
print('lower_year_limit:',lower," upper_year_limit:",upper)
df_cleaned = df_cleaned[lower<(df_cleaned[feature])]
print(f'Cleand year dataset : {df_cleaned.shape}')
print(f'Outliers year count : {len(df)-len(df_cleaned)}')

# Linear Plot
bins = range(0, int(df_cleaned['mileage'].max()) + 10000, 10000)# group mileage by every 10000km
df_cleaned['mileage_bin'] = pd.cut(df_cleaned['mileage'], bins=bins)

# calculate average price groupby mileage_bin
mileage_avg_price = df_cleaned.groupby('mileage_bin')['price(Georgian Lari)'].mean().reset_index()
mileage_avg_price['mileage_bin_str']=mileage_avg_price['mileage_bin'].astype(str)

# order the mileage
cat_type=CategoricalDtype(categories=mileage_avg_price['mileage_bin_str'],ordered=True)
mileage_avg_price['mileage_bin_str']=mileage_avg_price['mileage_bin_str'].astype(cat_type)

sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
sns.lineplot(data=mileage_avg_price,x='mileage_bin_str',y='price(Georgian Lari)',marker='o')
plt.title('Average Price by mileage')
plt.xlabel('Mileage Range(km)')
plt.ylabel('Average Price')
plt.xticks(rotation=45)

#KDE Plot - Kernel Density Estimation
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))

sns.kdeplot(data=df_cleaned, x='mileage',y='price(Georgian Lari)',cmap='Blues',fill=True,levels=10)
plt.title('KDE Plot of price vs mileage')
plt.xlabel('Mileage')
plt.ylabel('Average Price')
# plt.show()

# Histogram Plot
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
sns.histplot(df_cleaned['mileage'],bins=100,kde=True)
plt.title('Histogram (count plot for numerical data)')
# plt.show()

# Violine Plot
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
plot = sns.violinplot(data=df_cleaned,x='product_year',y='price(Georgian Lari)',inner='quartile')
plt.title('Violine Plot of product_year vs price')
plt.xlabel('product_year')
plt.ylabel('Price')
plt.xticks(rotation=90)
# plt.show()

# Box Plot
sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,8))
plot = sns.boxplot(data=df_cleaned,x='product_year',y='price(Georgian Lari)')
plt.title('Violine Plot of product_year vs price')
plt.xlabel('product_year')
plt.ylabel('Price')
plt.xticks(rotation=90)
# plt.show()

# Getting Unique values in categorical columns
unique_counts=df_cleaned[['fuel_type','gear','door_type','color','manufacture']].nunique().sort_values()
plt.figure(figsize=(20,8))
sns.barplot(x=unique_counts.index,y=unique_counts.values,palette='Set2')

plt.title('Unique Value Counts for Categorical Columns')
plt.xlabel('Categorical Columns')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()