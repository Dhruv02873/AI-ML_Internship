# AI-ML Internship Task-2: Exploratory Data Analysis (EDA)

## Objective
The goal of this task was to explore and understand the Titanic dataset using Python. Through visualizations and basic statistics, we aimed to uncover patterns and trends that could help in later machine learning tasks.

## Dataset
We're using the classic Titanic dataset from Kaggle:
🔗 Titanic - Machine Learning from Disaster

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Steps
1.  Generate summary statistics (mean, median, std, etc.).
2.  Create histograms and boxplots for numeric features.
3.  Use pairplot/correlation matrix for feature relationships.
4.  Identify patterns, trends, or anomalies in the data.
5.  Make basic feature-level inferences from visuals.

## Key Insights from the Data[Inferences]
- From describe():
The dataset has 891 rows (passengers).
The Age column is missing 177 values.
The Cabin column has too many missing values — it won’t be useful without serious cleanup.

- From Histograms:
Age is almost normally distributed but leans slightly to the right.
Fare is heavily skewed — most people paid low, but a few paid very high.
Pclass 3 has the most passengers — third-class was the most crowded.

- From Boxplots:
Survivors were generally younger.
There were more children among the survivors.
First-class passengers tended to be older, while third-class passengers were younger on average.

- From Pairplot:
Many survivors belonged to higher social classes (lower Pclass).
Survivors also had a tendency to pay higher fares.

- From pattern identification:
Fare is highly positively skewed.
Age has a slight positive skew.

## Files Included
- `main.py` – The main Python script for this task
- `Output/` – A folder containing all the plots generated during EDA


## How to Run This Project
1.  Make sure the Titanic dataset CSV (Titanic-Dataset.csv) is in the same folder.

2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the script:

   ```bash
   python first.py
