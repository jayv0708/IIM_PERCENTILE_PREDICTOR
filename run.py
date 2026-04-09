import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

print("\n--- IIM Target Percentile Predictor ---")
try:
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    targets = joblib.load("targets.pkl")
except:
    print("Model not found. Please run train_model.py first.")
    exit()

try:
    tenth = float(input("Enter 10th Percentage (0-100): "))
    twelfth = float(input("Enter 12th Percentage (0-100): "))
    ug = float(input("Enter Undergrad Percentage (0-100): "))
    
    print("\nStreams: Engineering, Commerce, Science, Arts")
    stream = input("Enter your Undergrad Stream exactly as above: ").strip()
    
    work_ex = float(input("Enter Work Experience in Months: "))
    
    print("\nCategories: General, OBC, SC, ST, EWS, PwD")
    category = input("Enter your category exactly as above: ").strip()
    
    print("\nGenders: Male, Female, Other")
    gender = input("Enter your gender exactly as above: ").strip()

    input_dict = {
        '10th_Percentage': [tenth],
        '12th_Percentage': [twelfth],
        'Undergrad_Percentage': [ug],
        'Work_Experience_Months': [work_ex]
    }
    
    for col in features:
        if col not in input_dict:
            input_dict[col] = [0]
            
    if f"Category_{category}" in input_dict: input_dict[f"Category_{category}"] = [1]
    if f"Gender_{gender}" in input_dict: input_dict[f"Gender_{gender}"] = [1]
    if f"Undergrad_Stream_{stream}" in input_dict: input_dict[f"Undergrad_Stream_{stream}"] = [1]

    input_df = pd.DataFrame(input_dict)[features]
    predictions = model.predict(input_df)[0]
    
    print(f"\n=> Profile: {gender}, {category}, {stream}, 10/12/UG: {tenth}/{twelfth}/{ug}, WorkEx: {work_ex}m")
    print("\n=> Estimated Target Percentiles to secure an interview:")
    for i, target in enumerate(targets):
        print(f"   - {target.replace('Target_', '')}: {predictions[i]:.2f} percentile")

except ValueError:
    print("\nInvalid input. Please enter numeric values where requested.")
