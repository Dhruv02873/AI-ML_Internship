import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression      
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv('Housing.csv')
print("Printing first 5 rows:\n",df.head())
print("\nMissing Value Count:\n",df.isnull().sum())
#since no missing value we only have to convert character data to numeric data for preprocessing
df_encodeing = pd.get_dummies(df, drop_first=True)

X = df_encodeing.drop("price", axis=1)
y = df_encodeing["price"]
#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#linear regression
model = LinearRegression()
model.fit(X_train, y_train)
#evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")

coefficients = pd.Series(model.coef_, index=X.columns)
# Plotting the coefficients
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh', color='skyblue')
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(True)
plt.tight_layout()
plt.show()