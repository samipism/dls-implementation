import streamlit as st

from DataLoader.Dataloader import Dataloader
from Algorithms.DL_Professional import DL_Professional
from Algorithms.DL_Stern import DL_Stern

target_score_dls = 0
target_score_dl_pro = 0
dls_par_score = 0
dl_pro_par_score = 0

dummy_t20_datapath = 'data/dummy_t20_data.csv'
dummy_odi_datapath = 'data/dummy_odi_data.csv'

data_ob = Dataloader(dummy_t20_datapath, dummy_odi_datapath)
dl_pro_ob = DL_Professional(data_ob)
dl_stern_ob = DL_Stern(data_ob)


with st.sidebar:
    st.title("Cricket Target Score Prediction")
    st.header("Input Parameters")
    team1_score = st.number_input("Team 1 Score", value=250)
    team2_wicktes_down = st.number_input("Team 2 Wickets Lost till now", value=2)
    team2_overs_down = st.number_input("Team 2 Overs Completed", value=20.0)
    lost_overs_due_to_rain = st.number_input("Team 2 overs lost due to Rain", value=5.0)
    if st.button("Calculate Target"):
        dls_par_score,target_score_dls = dl_stern_ob.inference_ODI(team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain)
        dl_pro_par_score,target_score_dl_pro = dl_pro_ob.inference_ODI(team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain)
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
