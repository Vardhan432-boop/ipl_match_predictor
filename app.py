import streamlit as st
import pandas as pd
import joblib

# Load saved model and label encoder
model = joblib.load("ipl_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="IPL Predictor", layout="centered")
st.title("IPL Match Winner Predictor")

teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
    'Punjab Kings', 'Rajasthan Royals', 'Gujarat Titans', 'Lucknow Super Giants'
]

venues = [
    'M Chinnaswamy Stadium',
    'Wankhede Stadium',
    'Eden Gardens',
    'MA Chidambaram Stadium',
    'Narendra Modi Stadium'
]

# Inputs
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options)

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
venue = st.selectbox("Venue", venues)
season = st.number_input("Season", min_value=2008, max_value=2026, value=2024)

def get_team_probabilities(input_df, selected_team1, selected_team2):
    """
    Returns probability assigned by the model to team1 and team2.
    """
    probs = model.predict_proba(input_df)[0]
    classes = model.named_steps["classifier"].classes_   # encoded labels
    decoded_classes = le.inverse_transform(classes)

    prob_map = {team: prob for team, prob in zip(decoded_classes, probs)}

    team1_prob = prob_map.get(selected_team1, 0.0)
    team2_prob = prob_map.get(selected_team2, 0.0)

    return team1_prob, team2_prob, prob_map

if st.button("Predict Winner"):
    input_df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "season": season
    }])

    pred_encoded = model.predict(input_df)[0]
    pred_team = le.inverse_transform([pred_encoded])[0]

    team1_prob, team2_prob, prob_map = get_team_probabilities(input_df, team1, team2)

    st.success(f"Predicted Winner: {pred_team}")

    st.subheader("Confidence between selected teams")
    conf_df = pd.DataFrame({
        "Team": [team1, team2],
        "Confidence": [team1_prob, team2_prob]
    }).set_index("Team")

    st.bar_chart(conf_df)

    st.write(f"{team1}: {team1_prob * 100:.2f}%")
    st.write(f"{team2}: {team2_prob * 100:.2f}%")

    st.subheader("Top 3 model predictions")
    top3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]
    for team, prob in top3:
        st.write(f"{team}: {prob * 100:.2f}%")