# Task-2-internship
# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Superstore.csv", encoding='latin1')

# Display the first few rows
print("Initial Data Preview:")
print(df.head())

# 1. Data Cleaning

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values
df.fillna({
    'Sales': df['Sales'].median(),
    'Profit': df['Profit'].median()
}, inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Detect and remove outliers using IQR (Interquartile Range) for Sales and Profit
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

df = remove_outliers_iqr(df, 'Sales')
df = remove_outliers_iqr(df, 'Profit')


# 2. Statistical Analysis

print("\nDescriptive Statistics:")
print(df[['Sales', 'Profit']].describe())



# Correlation matrix
print("\nCorrelation Matrix:")
print(df[['Sales', 'Profit']].corr())


# 3. Data Visualization

# Set plot style
sns.set(style="whitegrid")

# Histogram: Distribution of Sales
plt.figure(figsize=(10, 5))
sns.histplot(df['Sales'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Histogram: Distribution of Profit
plt.figure(figsize=(10, 5))
sns.histplot(df['Profit'], bins=30, kde=True, color='lightgreen')
plt.title('Distribution of Profit')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.show()

# Boxplot: Sales by Region
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Region', y='Sales', palette='pastel')
plt.title('Sales by Region')
plt.show()

# Boxplot: Profit by Product Category
if 'Category' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Category', y='Profit', palette='Set2')
    plt.title('Profit by Product Category')
    plt.show()

# Heatmap: Correlation between Sales and Profit
plt.figure(figsize=(6, 4))
sns.heatmap(df[['Sales', 'Profit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# END OF EDA

print("\nEDA Completed Successfully.")
