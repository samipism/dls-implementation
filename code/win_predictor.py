import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("../data/04_cricket_1999to2011.csv")

def preprocess_train(df):
    df['Overs.Remaining'] = 50 - df['Over']
    df['Wickets.Down'] = 10 - df['Wickets.in.Hand']

    df = df.copy()

    df = df[df['Innings']==2]
    df.loc[:,'Win.label'] = (df['At.Bat'] == df['Winning.Team']).astype(int)

    filtered_columns = ['Overs.Remaining', 'Wickets.in.Hand','Run.Rate','Run.Rate.Required', 'Win.label']
    df = df[filtered_columns]

    X_cols = ['Overs.Remaining', 'Wickets.in.Hand','Run.Rate','Run.Rate.Required']
    X = df[X_cols]
    y = df['Win.label']

    df_cleaned = df.dropna(subset=X_cols)

    X = df_cleaned[X_cols]
    y = df_cleaned['Win.label']

    return X,y

# Train-test split
X,y = preprocess_train(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model definition
model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

# Predict win proba
y_prob = model.predict_proba(X_test)[:, 1]  
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



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