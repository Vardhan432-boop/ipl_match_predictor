# IPL Match Winner Predictor 🏏

An end-to-end Machine Learning project that predicts the winner of an IPL match using historical IPL data and team statistics.

Built using:

* Python
* scikit-learn
* Streamlit
* Pandas

---

# Features 🚀

* Predict IPL match winner
* Team vs Team confidence comparison
* Interactive Streamlit UI
* One-hot encoded categorical features
* Random Forest based prediction model
* Match probability visualization
* Clean and simple interface

---

# Tech Stack 🛠️

* Python
* Pandas
* NumPy
* scikit-learn
* Streamlit
* Joblib

---

# Dataset 📊

Dataset used:

## [IPL Complete Dataset (2008–2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?utm_source=chatgpt.com)

Main file used:

```text id="6vjlwm"
matches.csv
```

---

# ML Workflow ⚙️

## Data Preprocessing

* Removed missing values
* Removed old/defunct IPL teams
* Filtered modern IPL seasons
* Encoded categorical features

## Feature Engineering

Features used:

* Team 1
* Team 2
* Venue
* Toss Winner
* Toss Decision
* Season
* Head-to-head statistics

## Model Training

Models experimented with:

* Logistic Regression
* Decision Tree
* Random Forest

Final model selected:

## Random Forest Classifier

---

# Project Structure 📁


ipl-match-predictor/
│
├── app.py
├── train.py
└── README.md
```

---

# Installation 🔧

Clone the repository:

```bash
git clone <your-repo-link>
cd ipl-match-predictor
```

---

# Run the Project ▶️

Train the model:

```bash
python train.py
```

Start Streamlit app:

```bash
streamlit run app.py
```

---


# Future Improvements 🔥

* Live IPL API integration
* Player-based analysis
* Recent form prediction
* Advanced feature engineering
* XGBoost integration
* Ball-by-ball win probability

---

# Learning Outcomes 📚

This project helped in understanding:

* Data preprocessing
* Feature engineering
* Classification models
* Model deployment
* Streamlit frontend integration
* End-to-end ML pipelines

---

# Author ✨

Developed by Vardhan.
