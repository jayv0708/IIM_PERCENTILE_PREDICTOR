import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import sys
import subprocess

st.set_page_config(page_title="IIM Target Percentile Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model_v3():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    targets = joblib.load("targets.pkl")
    return model, features, targets

@st.cache_data
def load_cutoffs():
    college_csv_dir = "cat data/College_wise_category_cutoff"
    CAT_MAP = {
        'NC-OBC-cum-Transgender': 'OBC', 'NC-OBC': 'OBC',
        'PWD': 'PwD', 'SC': 'SC', 'ST': 'ST',
        'General': 'General', 'EWS': 'EWS'
    }
    rules = {}
    for file in os.listdir(college_csv_dir):
        if file.endswith('.csv'):
            college_name = file.replace('.csv', '').capitalize()
            if 'Calcutta' in college_name: college_name = 'IIM Calcutta'
            elif not college_name.startswith('Iim'): college_name = 'IIM ' + college_name
            else: college_name = college_name.replace('Iim', 'IIM')
            try:
                df = pd.read_csv(os.path.join(college_csv_dir, file), header=1)
                df = df.dropna(subset=['Category'])
                df = df[~df['Category'].astype(str).str.contains('Selection', na=False)]
                df['Category'] = df['Category'].replace(CAT_MAP)
                df = df[df['Category'].isin(CAT_MAP.values())]
                rules[college_name] = df
            except Exception:
                pass
    return rules

if not os.path.exists("model.pkl"):
    st.info("⚙️ Model missing! Automatically triggering a local retraining sequence optimized for this server... this will take about 15 seconds.")
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        st.success("✅ Training complete! Reloading resources...")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Failed to train model automatically: {e}")
        st.stop()

try:
    model, features, targets = load_model_v3()
    cutoffs = load_cutoffs()
except FileNotFoundError:
    st.error("Model files not found! Please run `python train_model.py` first.")
    st.stop()

st.title("🎓 Target Percentile Predictor")
st.markdown("Enter your complete profile below to predict the **Target CAT Percentile** you need to get an interview call from each IIM.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Academic Profile")
    tenth  = st.number_input("10th Percentage",  min_value=0.0, max_value=100.0, value=90.0)
    twelfth = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=85.0)
    ug = st.number_input("Undergrad Percentage", min_value=0.0, max_value=100.0, value=80.0)
    stream = st.selectbox("Undergrad Stream", ["Engineering", "Commerce", "Science", "Arts"])

with col2:
    st.subheader("🧑‍💼 Work Exp & Diversity")
    work_ex = st.number_input("Work Experience (Months)", min_value=0, max_value=120, value=12)
    category = st.selectbox("Reservation Category", ["General", "OBC", "SC", "ST", "EWS", "PwD"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.divider()

if st.button("🔍 Evaluate Target Percentiles", use_container_width=True):
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
    
    pred_map = {targets[i].replace('Target_', ''): float(predictions[i]) for i in range(len(targets))}
    
    results = []
    for college, cutoff_df in sorted(cutoffs.items()):
        cat_row = cutoff_df[cutoff_df['Category'] == category]
        if cat_row.empty:
            cat_row = cutoff_df[cutoff_df['Category'] == 'General']
        if cat_row.empty:
            continue
        try:
            v_cut = float(cat_row['VARC'].values[0])
            d_cut = float(cat_row['DI & LR'].values[0])
            q_cut = float(cat_row['QA'].values[0])
            o_min = float(cat_row['Overall Percentile'].values[0])
            target_perc = pred_map.get(college, 99.0)
            
            # Map NAs if cutoffs are very high or zero
            if o_min == 0:
                continue
                
            results.append({
                "Top B-Schools": college,
                "VARC %ile Cutoff": "NA" if v_cut == 0 else f"{v_cut:.2f}",
                "DILR %ile Cutoff": "NA" if d_cut == 0 else f"{d_cut:.2f}",
                "QA %ile Cutoff": "NA" if q_cut == 0 else f"{q_cut:.2f}",
                "Overall Min %ile Cutoff": "NA" if o_min == 0 else f"{o_min:.2f}",
                "Target Percentile": f"{target_perc:.2f}"
            })
        except Exception:
            continue
            
    if results:
        res_df = pd.DataFrame(results).set_index("Top B-Schools")
        st.markdown("### Profile Evaluation")
        
        def highlight_target(val):
            return 'background-color: #66c2a5; color: black; font-weight: bold'
            
        styled_df = res_df.style.applymap(highlight_target, subset=['Target Percentile'])
        st.dataframe(styled_df, use_container_width=True, height=600)
    else:
        st.error("Could not evaluate profile for the selected category.")
