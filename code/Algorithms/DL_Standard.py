import numpy as np
from scipy.optimize import minimize


class DL_Standard: 

    def __init__(self, data_ob, G_50=2.045e+02, b=3.159e-02, a1=-1.196e-01, a2=-1.748e-03,  a3=3.840e-04):
        self.G_50 = G_50
        self.b = b
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.data = self.filter_data(data_ob)


    def filter_data(self, data_ob):
        data = data_ob.odi_data.copy()
        data = data[(data['season'] >= 2000) & (data['season'] <= 2012)]
        return data 


    def calculate_z(self, u, w):
        Z_0 = self.G_50/(1 - np.exp(-50*self.b))
        F_w = 1 + self.a1*w + self.a2 * (w**2) + self.a3 * (w**3)
        Z_ = Z_0 * F_w * (1 - np.exp((-u*self.b)/F_w))
        return Z_
    

    def loss_function(self, params):

        G_50, b, a1, a2, a3 = params
        Z_0 = G_50 / (1 - np.exp(-50 * b))
        
        self.data['F_w'] = 1 + a1 * self.data['wickets_down'] + a2 * (self.data['wickets_down']**2) + a3 * (self.data['wickets_down']**3)
        self.data['Z_std'] = Z_0 * self.data['F_w'] * (1 - np.exp(-b * self.data['balls_remaining'] / 6 / self.data['F_w']))
        
        self.data['loss'] = (self.data['runs_remaining'] - self.data['Z_std'])**2
        
        return self.data['loss'].mean()


    def optimize(self):

        #initial_guess = [2.045e+02, 3.159e-02, -1.196e-01, -1.748e-03,  3.840e-04]

        initial_guess = [200, 0.1, -0.1, -0.01, 0.001]
        print(f"{initial_guess = }")

        result = minimize(self.loss_function, initial_guess, method='Nelder-Mead')
        
        self.G_50, self.b, self.a1, self.a2, self.a3 = result.x 
        
        print("Optimization Result:", result)


    def inference_ODI(self, team1_score, team2_wickets_down, team2_overs_completed, lost_overs_due_to_rain): 

        u_1 = 50 - team2_overs_completed
        u_2 = 50 - (team2_overs_completed + lost_overs_due_to_rain)

        P_u1_w = self.calculate_z(u_1, team2_wickets_down)/self.calculate_z(50, 0)
        P_u2_w = self.calculate_z(u_2, team2_wickets_down)/self.calculate_z(50, 0)

        R_1 = 1 * 100
        R_2 = (1 - P_u1_w + P_u2_w)*100

        team2_target = None

        if R_2 <= R_1:
            team2_target = team1_score * R_2/R_1
        else:
            team2_target = team1_score + self.G_50*(R_2 - R_1)

        return team2_target, int(np.ceil(team2_target))
    

    def inference_T20I(self, team1_score, team2_wickets_down, team2_overs_completed, lost_overs_due_to_rain): 
        initial_run = self.calculate_z(30, 0)
        new_team1_score = initial_run + team1_score
        new_team2_overs_completed = 30 + team2_overs_completed
        predicted_par_ODI, _ = self.inference_ODI(new_team1_score, team2_wickets_down, new_team2_overs_completed, lost_overs_due_to_rain)

        return (predicted_par_ODI - initial_run), int(np.ceil(predicted_par_ODI - initial_run))


    





    