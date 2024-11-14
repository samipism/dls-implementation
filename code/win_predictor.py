import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

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

with open('model.pkl','wb') as f:
    pickle.dump(model,f)



