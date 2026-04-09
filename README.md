# IIM Target Percentile Predictor

## Overview

The IIM Target Percentile Predictor is a machine learning application designed to estimate the minimum CAT percentile required for a candidate to receive interview calls from all 19 Indian Institutes of Management (IIMs).

Unlike traditional admission predictors that provide binary outcomes, this system outputs precise percentile targets for each IIM based on a candidate’s academic profile, work experience, and demographic attributes.

---

## Features

* Predicts target CAT percentile for 19 IIMs simultaneously
* Uses a Multi-Output Random Forest Regressor
* Incorporates:

  * Academic performance impact
  * Category-based cutoff adjustments
  * Gender and academic diversity factors
* Provides two interfaces:

  * Streamlit web application
  * Command-line interface (CLI)

---

## Model Details

* Algorithm: Multi-Output Random Forest Regressor
* Training Data: Approximately 15,000 synthetically generated profiles
* Output: 19 regression targets (one for each IIM)

The model captures non-linear relationships between academic scores, work experience, and admission criteria.

---

## Dataset Engineering

Due to the absence of publicly available admission datasets, this project generates a synthetic dataset that approximates real-world IIM selection behavior.

### Input Features

* 10th Percentage
* 12th Percentage
* Undergraduate Percentage
* Work Experience (months)
* Undergraduate Stream (Engineering, Commerce, Science, Arts)
* Category (General, OBC, SC, ST, EWS, PwD)
* Gender (Male, Female, Other)

### Data Logic

* Baseline percentiles mapped per IIM and category
* Academic penalties applied for lower scores
* Diversity bonuses for non-engineers and female candidates
* Generates target percentile for each IIM

---

## Project Structure

```bash
IIM_PERCENTILE_PREDICTOR/
│
├── build_iim_dataset.py     # Synthetic dataset generator
├── train_model.py           # Model training script
├── app.py                   # Streamlit web app
├── run.py                   # CLI interface
├── model.pkl                # Trained model
├── features.pkl             # Feature encoder
├── iim_dataset.csv          # Generated dataset
├── requirements.txt         # Dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/IIM_PERCENTILE_PREDICTOR.git
cd IIM_PERCENTILE_PREDICTOR
pip install -r requirements.txt
```

---

## Usage

### Run Web Application

```bash
streamlit run app.py
```

Access the application at:

```
http://localhost:8501
```

---

### Run CLI

```bash
python run.py
```

---

### Optional: Generate Dataset

```bash
python build_iim_dataset.py
```

---

### Optional: Train Model

```bash
python train_model.py
```

---

## Sample Output

| IIM           | Target Percentile |
| ------------- | ----------------- |
| IIM Ahmedabad | 99.85             |
| IIM Bangalore | 99.70             |
| IIM Calcutta  | 99.60             |
| ...           | ...               |

---

## Disclaimer

This project uses synthetic data modeled on real-world trends. Predictions are estimates and not guarantees of admission. Final selection depends on additional factors such as personal interviews, written ability tests, and institute-specific criteria.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

## Future Work

* Integration with real admission datasets
* Exploration of advanced models such as XGBoost and LightGBM
* Resume and profile scoring system
* Deployment as a public web application

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes or submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as a data-driven tool to help CAT aspirants set realistic percentile targets and optimize preparation strategy.
