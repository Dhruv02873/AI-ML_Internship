import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
#STEP-1-Importing the dataset and Displaying basic information about the dataset
df=pd.read_csv('Titanic-Dataset.csv')
df.info()
df.head()
df.describe()
df.isnull().sum()
#STEP-2-useing mode median or mean to fill the missing values and even drop a column if it has too many missing values(Such info was found above in the isnull() function)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
#STEP-3-Converting categorical data to numerical data(because ML models work with numerical data)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
#STEP-4-Scaling the data
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
#STEP-5-Outlier detection and removal
sns.boxplot(x=df['Fare'])
plt.show()
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]
#STEP-6-Visualizing the cleaned data
df.to_csv("cleaned_titanic.csv", index=False)




