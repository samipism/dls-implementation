import streamlit as st
from inference import inference_ODI
# from win_prob import win_predictor

target_score_dls = 0
target_score_dl_pro = 0
dls_par_score = 0
dl_pro_par_score = 0

Z_dls_params = [3.216e-01 , -4.633e+00,  3.518e-02 , -2.623e-01]
Z_std_params = [2.045e+02, 3.159e-02, -1.196e-01, -1.748e-03,  3.840e-04]


with st.sidebar:
    st.title("Cricket Target Score Prediction")
    st.header("Input Parameters")
    team1_score = st.number_input("Team 1 Score", value=250)
    team2_wicktes_down = st.number_input("Team 2 Wickets Lost till now", value=2)
    team2_overs_down = st.number_input("Team 2 Overs Completed", value=20.0)
    lost_overs_due_to_rain = st.number_input("Team 2 overs lost due to Rain", value=5.0)
    if st.button("Calculate Target"):
        dls_par_score,target_score_dls = inference_ODI(Z_dls_params,Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, method='DLS')
        dl_pro_par_score,target_score_dl_pro = inference_ODI(Z_dls_params,Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, method='DL-PRO')
        dls_par_score = format(dls_par_score,".2f")
        dl_pro_par_score = format(dl_pro_par_score,".2f")
        
col1, col2 = st.columns(2)
with col1:
    ct = st.container(border=True)
    
    ct.write('''
            # Duckworth Lewis Stern Method
            ''')
    ct.write(f"Predicted Target for Team 2: {target_score_dls}")
    ct.write(f"Par Score for Team 2: {dls_par_score}")
with col2:
    ct2 = st.container(border=True)
    ct2.write('''
            # Duckworth Lewis Pro Edition
            ''')
    ct2.write(f"Predicted Target for Team 2: {target_score_dl_pro}") 
    ct2.write(f"Par Score for Team 2: {dl_pro_par_score}")

# with tab2:
#     team2_score = st.number_input("Team 2 Current Score", value=2)
    
#     st.write('''
#             # Win Predictor Logistic Regression
#             ''')
#     prob = win_predictor(team1_score, team2_wicktes_down, team2_overs_down, team2_score)
#     st.success(f"Win probability for Team 2: {prob}")