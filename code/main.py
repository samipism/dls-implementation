
from DataLoader.Dataloader import Dataloader
from Algorithms.DL_Standard import DL_Standard
from Algorithms.DL_Professional import DL_Professional
from Algorithms.DL_Stern import DL_Stern
from Utils.Plot import Graph

if __name__ == '__main__':



    dummy_t20_datapath = '../data/dummy_t20_data.csv'
    dummy_odi_datapath = '../data/dummy_odi_data.csv'

    data_ob = Dataloader(dummy_t20_datapath, dummy_odi_datapath)
    dl_pro_ob = DL_Professional(data_ob)
    dl_stern_ob = DL_Stern(data_ob)


    # data_ob = Dataloader()

    # dl_std = DL_Standard(data_ob)
    # dl_pro = DL_Professional(data_ob)
    # dl_stern = DL_Stern(data_ob)

    # graph = Graph(data_ob, dl_std, dl_pro, dl_stern)

    # graph.plot_observed_vs_predicted()
    # graph.plot_resource_remaining()


    # # ODI Scenario
    # print("\nODI Scenario: ")
    # team1_score = 350
    # team2_wicktes_down = 2
    # team2_overs_completed = 10
    # lost_overs_due_to_rain = 5


    # print("\nDL-STD: ",dl_std.inference_ODI( team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    # print("\nDL-PRO: ",dl_pro.inference_ODI( team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    # print("\nDLS: ",dl_stern.inference_ODI( team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    
    # #T20I scenario
    # print("\nT20I Scenario: ")
    # team1_score = 139
    # team2_wicktes_down = 2
    # team2_overs_completed = 6.5
    # lost_overs_due_to_rain = 13.1
    # print("\nDL-STD: ",dl_std.inference_T20I(team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    # print("\nDL-PRO: ",dl_pro.inference_T20I(team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    # print("\nDLS: ",dl_stern.inference_T20I(team1_score, team2_wicktes_down, team2_overs_completed, lost_overs_due_to_rain))
    


