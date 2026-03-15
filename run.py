import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Loading data...")
# Loading the dataset
df = pd.read_csv('admission_predict.csv')

# Renaming the columns with appropriate names
df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'Probability'})

# Removing the serial no, column
df.drop('Serial No.', axis='columns', inplace=True)

# Replacing the 0 values from ['GRE','TOEFL','University Rating','SOP','LOR','CGPA'] by NaN
df_copy = df.copy(deep=True)
df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']] = df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']].replace(0, np.NaN)

# Splitting the dataset in features and label
X = df_copy.drop('Probability', axis='columns')
y = df_copy['Probability']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model Accuracy (R^2 Score): {score*100:.2f}%")

print("\n--- Interactive Prediction ---")
print("Please provide the following details to predict your admission chances:")
try:
    gre = float(input("Enter GRE Score (e.g., 320): "))
    toefl = float(input("Enter TOEFL Score (e.g., 110): "))
    rating = float(input("Enter University Rating (1-5): "))
    sop = float(input("Enter SOP Strength (1.0-5.0): "))
    lor = float(input("Enter LOR Strength (1.0-5.0): "))
    cgpa = float(input("Enter CGPA (e.g., 8.5): "))
    research = float(input("Enter Research Experience (1 for Yes, 0 for No): "))

    pred = round(model.predict([[gre, toefl, rating, sop, lor, cgpa, research]])[0]*100, 3)
    print(f'\n=> Your estimated chance of admission is {pred}%')
except ValueError:
    print("\nInvalid input. Please enter numeric values.")
