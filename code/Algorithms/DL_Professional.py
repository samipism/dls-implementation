import numpy as np
from scipy.optimize import root_scalar

from Algorithms.DL_Standard import DL_Standard


class DL_Professional(DL_Standard):

    def calculate_z(self, u, w, lmda):

        n0 = 5
        Z_0 = self.G_50/(1 - np.exp(-50*self.b))
        F_w = 1 + self.a1*w + self.a2 * (w**2) + self.a3 * (w**3)

        n_w = n0 * F_w

        return Z_0 * F_w * (lmda**(n_w+1)) * (1 - np.exp((-u*self.b)/(F_w * (lmda**n_w))))


    def calculate_lambda(self, team1_score):
        u, w = 50, 0
        solution = root_scalar(lambda x: self.calculate_z(u, w, x) - team1_score, bracket=[1,100])

        return solution.root
    

    def inference_ODI(self, team1_score, team2_wickets_down, team2_overs_completed, lost_overs_due_to_rain): 

        u_1 = 50 - team2_overs_completed
        u_2 = 50 - (team2_overs_completed + lost_overs_due_to_rain)

        lmda = self.calculate_lambda(team1_score)

        P_u1_w = self.calculate_z(u_1, team2_wickets_down, lmda)/self.calculate_z(50, 0, lmda)
        P_u2_w = self.calculate_z(u_2, team2_wickets_down, lmda)/self.calculate_z(50, 0, lmda)

        R_1 = 1 * 100
        R_2 = (1 - P_u1_w + P_u2_w)*100

        team2_target = None

        if R_2 <= R_1:
            team2_target = team1_score * R_2/R_1
        else:
            team2_target = team1_score + self.G_50*(R_2 - R_1)

        return team2_target, int(np.ceil(team2_target))
    

    def inference_T20I(self, team1_score, team2_wickets_down, team2_overs_completed, lost_overs_due_to_rain): 
        initial_run = self.calculate_z(30, 0, 1)
        new_team1_score = initial_run + team1_score
        new_team2_overs_completed = 30 + team2_overs_completed
        predicted_par_ODI, _ = self.inference_ODI(new_team1_score, team2_wickets_down, new_team2_overs_completed, lost_overs_due_to_rain)

        return (predicted_par_ODI - initial_run), int(np.ceil(predicted_par_ODI - initial_run))

    


