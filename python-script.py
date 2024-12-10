#TASK 1

import pandas as pd

# Load the dataset (for example, using the Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Display the first few rows
df.head()

# Explore the structure
df.info()

# Check for missing values
df.isnull().sum()

# In case of missing data, fill or drop
# Example: df.fillna(0, inplace=True) or df.dropna(inplace=True)

#TASK 2

# Basic statistics for numerical columns
df.describe()

# Grouping by species and computing the mean of each numerical column
df.groupby('species').mean()

# Example: Identifying the average sepal length per species
df.groupby('species')['sepal length (cm)'].mean()


#TASK 3

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart (example, assuming you have time-series data)
# plt.plot(time_data, value_data)
# plt.title('Trend Over Time')
# plt.xlabel('Time')
# plt.ylabel('Value')

# Bar chart (average petal length per species)
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram (distribution of sepal length)
plt.hist(df['sepal length (cm)'], bins=10, edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot (sepal length vs. petal length)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()


#TASK 4

try:
    # Load dataset (replace with your actual dataset path)
    df = pd.read_csv('path_to_your_file.csv')
except FileNotFoundError:
    print("The file was not found. Please check the path.")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
