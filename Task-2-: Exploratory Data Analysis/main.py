import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Titanic-Dataset.csv')

# Step-1 we had to display summary statistics like mean, meadian, so we used describe to show summary stats  together  for all columns
#we could have also used df.mean(), df.median() etc. to get individual stats but that would be too long
t= df.describe(include='all')
print(t)
#Step-2 we have to display histogram and boxplot for numeric data
df.hist(figsize=(12, 8), bins=24, edgecolor='black')
plt.tight_layout()
plt.show()
#now code for boxplot, as there are many,i am just showing some of them
plt.figure(figsize=(12,8))
sns.boxplot(x='Survived'  ,y='Age'  ,data=df)
plt.title('Survived vs Age Boxplot')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass'  ,y='Age'  ,data=df)
plt.title('Pclass vs Age Boxplot')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x='Age'  ,y='Sex'  ,data=df)
plt.title('Sex vs Age Boxplot')
plt.show()
#Step-3 Pairplot matrix [easy to implement as compared to correlation matrix]
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()
#Step-4 Identify patterns, trends, or anomalies in the data/ Detecting Skewness
check=df.skew(numeric_only=True)
print("Skewness in the data:")
print(check)
#Step-5 Showing Inferences
# Inference from describe():
# - The dataset contains 891 rows.
# - Age column has 177 missing values .
# - Cabin has many missing values and is not  useful without preprocessing.
# Inference from histograms:
# - Age distribution is close to normal but slightly right-skewed.
# - Fare is heavily right-skewed; many passengers paid low fare, few paid high fare.
# - Most passengers are in Pclass 3.
# Inference from boxplots::
# - Survivors tend to be slightly younger on average.
# - More children are visible among survivors.
# - 1st class passengers are generally older.
# - 3rd class passengers are younger on average.
# Inference from Pairplot:
# - Survivors often belong to lower Pclass (higher status) and higher Fare.
# Inference from skewness:
# - Fare is highly positively skewed.
# - Age has slight positive skew.





