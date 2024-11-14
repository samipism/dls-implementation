import joblib
import pandas as pd

model = joblib.load("model.joblib")

def win_predictor(team1_score, team2_wickets_down, team2_overs_completed, team2_score):
    # team1_score = 309
    # team2_wicktes_down = 2
    # team2_overs_completed = 20
    # lost_overs_due_to_rain = 10
    # team2_score = 165

    # Input Mode-1 for checking if it works
    new_data = pd.DataFrame({
        'Overs.Remaining': [50-team2_overs_completed],
        'Wickets.in.Hand': [10-team2_wickets_down],
        'Run.Rate': [team2_score/team2_overs_completed],
        'Run.Rate.Required': [(team1_score - team2_score)/(50-team2_overs_completed)]
    })

    # Predict win probability
    win_probability = model.predict_proba(new_data)[:, 1]
    return win_probability * 100
    # print(f"Predicted Win Probability: {win_probability[0]:.2f}")

a = win_predictor(309,2,20,165)
print(a)