import streamlit as st
from inference import inference_ODI

target_score_dls = 0
target_score_dl_pro = 0
Z_dls_params = [3.216e-01 , -4.633e+00,  3.518e-02 , -2.623e-01]
Z_std_params = [3.216e-01 , -4.633e+00,  3.518e-02 , -2.623e-01]
# Z_dls_params = [ 1.471e+00 , 5.794e-01 , 4.691e-02 ,-1.326e+00]
# Z_std_params = [ 2.609e+02,  1.204e-02, -1.252e-01,  2.624e-03,  1.758e-04]

with st.sidebar:
    st.title("Cricket Target Score Prediction")
    st.header("Input Parameters")
    team1_score = st.number_input("Team 1 Score", value=250)
    team2_wicktes_down = st.number_input("Team 2 Wickets Down", value=2)
    team2_overs_down = st.number_input("Team 2 Overs Down", value=20)
    lost_overs_due_to_rain = st.number_input("Lost Overs Due to Rain", value=5)
    if st.button("Calculate Target"):
        target_score_dls = inference_ODI(Z_dls_params,Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, method='DLS')

st.write('''
         # Duckworth Lewis Stern Method
         ''')
st.success(f"Predicted Target Score for Team 2: {target_score_dls}")
