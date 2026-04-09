import os
import pandas as pd
import numpy as np

college_csv_dir = "cat data/College_wise_category_cutoff"
CAT_MAP = {
    'NC-OBC-cum-Transgender': 'OBC', 'NC-OBC': 'OBC',
    'PWD': 'PwD', 'SC': 'SC', 'ST': 'ST',
    'General': 'General', 'EWS': 'EWS'
}

rules = {}
for file in os.listdir(college_csv_dir):
    if file.endswith('.csv'):
        college = file.replace('.csv', '').replace('calcutta', 'Calcutta').strip()
        college = 'IIM ' + college.capitalize()
        try:
            df = pd.read_csv(os.path.join(college_csv_dir, file), header=1)
            df = df.dropna(subset=['Category'])
            df = df[~df['Category'].astype(str).str.contains('Selection', na=False)]
            df['Category'] = df['Category'].replace(CAT_MAP)
            df = df[df['Category'].isin(CAT_MAP.values())]
            rules[college] = df
        except Exception:
            pass

num_samples = 15000
np.random.seed(42)

categories = ['General', 'OBC', 'SC', 'ST', 'EWS', 'PwD']
category_probs = [0.35, 0.27, 0.15, 0.10, 0.10, 0.03]
cats = np.random.choice(categories, num_samples, p=category_probs)

streams = ['Engineering', 'Commerce', 'Science', 'Arts']
stream_probs = [0.60, 0.20, 0.10, 0.10]
streams_arr = np.random.choice(streams, num_samples, p=stream_probs)

genders = ['Male', 'Female', 'Other']
gender_probs = [0.65, 0.33, 0.02]
genders_arr = np.random.choice(genders, num_samples, p=gender_probs)

df = pd.DataFrame({
    'Category': cats,
    'Gender': genders_arr,
    '10th_Percentage': np.clip(np.random.normal(82, 9, num_samples), 45, 100),
    '12th_Percentage': np.clip(np.random.normal(80, 10, num_samples), 45, 100),
    'Undergrad_Percentage': np.clip(np.random.normal(75, 11, num_samples), 45, 100),
    'Undergrad_Stream': streams_arr,
    'Work_Experience_Months': np.clip(np.random.gamma(2.5, 10, num_samples), 0, 84).astype(int)
})

IIM_NAMES = sorted(list(rules.keys()))
print("Generating realistic target percentiles for specific IIMs...")

# Precise Base Percentiles for a General Male Engineer (90/90/80 profile)
GEN_BASES = {
    'IIM Ahmedabad': 99.85,
    'IIM Bangalore': 99.80,
    'IIM Calcutta': 99.75,
    'IIM Lucknow': 99.85,
    'IIM Indore': 99.00,
    'IIM Kozhikode': 99.50,
    'IIM Mumbai': 97.50,
    'IIM Shillong': 98.00,
    'IIM Rohtak': 97.00,
    'IIM Ranchi': 96.50,
    'IIM Raipur': 96.00,
    'IIM Trichy': 96.00,
    'IIM Kashipur': 95.50,
    'IIM Udaipur': 95.50
}

CATEGORY_OFFSETS = {
    'General': 0.0,
    'EWS': -2.5,
    'OBC': -4.5,
    'SC': -14.0,
    'ST': -25.0,
    'PwD': -25.0
}

cutoff_lookup = {}
for college in IIM_NAMES:
    cutoff_lookup[college] = {}
    for _, row in rules[college].iterrows():
        try:
            cutoff_lookup[college][row['Category']] = float(row['Overall Percentile'])
        except:
            pass

for college in IIM_NAMES:
    targets = []
    
    for _, student in df.iterrows():
        cat = student['Category']
        o_min = cutoff_lookup[college].get(cat, cutoff_lookup[college].get('General'))
        
        if o_min is None or o_min == 0:
            targets.append(99.0)
            continue
            
        base_target = GEN_BASES.get(college, 94.0)
        base_target += CATEGORY_OFFSETS.get(cat, 0.0)
        
        # Adjust based on academics
        acad_penalties = 0.0
        if student['10th_Percentage'] < 90:
            acad_penalties += (90 - student['10th_Percentage']) * 0.1
        if student['12th_Percentage'] < 90:
            acad_penalties += (90 - student['12th_Percentage']) * 0.1
        if student['Undergrad_Percentage'] < 80:
            acad_penalties += (80 - student['Undergrad_Percentage']) * 0.15
            
        # Adjust based on Diversity
        diversity_bonus = 0.0
        if student['Undergrad_Stream'] != 'Engineering':
            diversity_bonus += 1.5 if base_target > 97 else 2.0
        if student['Gender'] in ['Female', 'Other']:
            diversity_bonus += 1.5 if base_target > 97 else 2.0
            
        final_target = base_target + acad_penalties - diversity_bonus
        
        # Absolute minimum constraint
        final_target = max(final_target, o_min)
        # Add a tiny bit of random noise for ML mapping
        final_target += np.random.normal(0, 0.05)
        final_target = min(99.99, final_target)
        
        targets.append(final_target)
            
    df[f'Target_{college}'] = np.round(targets, 2)

df.to_csv("iim_admission_predict.csv", index=False)
print("Saved realistic synthetic target dataset.")
