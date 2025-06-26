# AI-ML Internship Task-3: Linear Regression

## Objective
The aim of this task is to implement and understand Simple and Multiple Linear Regression using a housing price dataset. The model is trained to predict house prices based on various features such as area, furnishing status, number of bedrooms, etc.

## Dataset
Housing Price Prediction Dataset  
Source: https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction  
File used: Housing.csv

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Key Concepts Practiced
1.Import and preprocess the dataset.
2.Split data into train-test sets.
3.Fit a Linear Regression model using sklearn.linear_model.
4.Evaluate model using MAE, MSE, R².
5.Plot regression line and interpret coefficients.


## Steps Performed

1. Data Loading
   - Loaded Housing.csv using pandas
   - Displayed the first five rows
   - Checked for missing values (none found)

2. Data Preprocessing
   - Converted categorical variables into numeric using `pd.get_dummies()`
   - Used `drop_first=True` to avoid multicollinearity (dummy variable trap)

3. Feature Selection
   - X (features): all columns except 'price'
   - y (target): the 'price' column

4. Data Splitting
   - Split the dataset into training and testing sets using an 80-20 ratio

5. Model Training
   - Created a Linear Regression model using `LinearRegression()` from `sklearn.linear_model`
   - Trained it on the training data

6. Model Evaluation
   - Used the test set to generate predictions
   - Calculated Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score
   - Printed the evaluation metrics

7. Feature Importance Visualization
   - Retrieved the model's learned coefficients
   - Plotted them using a horizontal bar chart to show feature impact on house prices

## Files Included
- `Housing.csv` - Dataset
- `main.py` - Main python script
- `Output` - output folder
## How to run this project
1. Make sure Housing.csv is in same folder as main.py
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn sklearn scikit-learn
   ```
3. Run the script:
   ```bash
   python main.py



