# AI-ML Internship Task-4: Classification with Logistic Regression

## Objective
The goal of this task is to build a binary classification model using logistic regression. The model predicts whether a tumor is malignant or benign based on features from the Breast Cancer Wisconsin dataset.

## Dataset
- Source: UCI Breast Cancer Wisconsin Diagnostic Dataset
- File used: data.csv
- Target column: diagnosis (M = Malignant, B = Benign)

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Hints:
1.Choose a binary classification dataset.
2.Train/test split and standardize features.
3.Fit a Logistic Regression model.
4.Evaluate with confusion matrix, precision, recall, ROC-AUC.
5.Tune threshold and explain sigmoid function.


## Steps Performed

1. Data Loading
   - Loaded data from data.csv using pandas
   - Checked for null values and structure

2. Preprocessing
   - Converted diagnosis labels from 'M' and 'B' to 1 and 0
   - Dropped the 'id' column as it is not useful for prediction
   - Scaled features using StandardScaler

3. Splitting the Data
   - Separated features and target
   - Split data into training and testing sets using 80-20 ratio

4. Evaluation
   - Generated predictions on test data
   - Calculated Confusion Matrix, Classification Report
   - Computed ROC-AUC score
   - Plotted ROC Curve for visual evaluation

## Files Included
- `main.py ` - main python script
- `data.csv` - dataset
- `Output` - Output Folder

## Run the script
1. Make sure dataset is in the same folder.
2. Install all required python libraries.
  ```bash
  pip install pandas numpy matplotlib seaborn sklearn
```
4. Run the python script
  ```bash
  python main.py
```
