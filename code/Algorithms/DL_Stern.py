
import numpy as np
from scipy.optimize import minimize

from Algorithms.DL_Professional import DL_Professional 


class DL_Stern(DL_Professional):

    def __init__(self, data_ob, c1=3.216e-01 , c2=-4.633e+00,  c3=3.518e-02 , c4=-2.623e-01):
        self.c1 = c1 
        self.c2 = c2 
        self.c3 = c3 
        self.c4 = c4
        super().__init__(data_ob)

    def filter_data(self, data_ob):
        data = data_ob.odi_data.copy()
        data = data[(data['season'] >= 2010) & (data['season'] <= 2015)]
        data = data[(data['total_runs'] > 300) | (data['target_score'] >300) ]
        data = data.copy()
        runs = np.where(data['innings'] == 1,data['total_runs'], data['target_score'] - 1)  
        lambdas = []
        for i in runs:
            lambdas.append(self.calculate_lambda(i))
        data['match_factor'] = np.array(lambdas)
        return data

    
    def calculate_z(self, u, w, lmda):
        
        Z_0 = self.G_50/(1-np.exp(-50*self.b))
        F_w = 1 + self.a1*w + self.a2*(w**2) + self.a3*(w**3)

        n0 = 5
        n_w = n0 * F_w

        alpha = -1 / (1+self.c1*(lmda-1)*np.exp(-self.c2*(lmda-1)))
        beta = -self.c3*(lmda-1)*np.exp(-self.c4*(lmda-1))

        g_u_lmda = 0
        if u != 0:
            g_u_lmda = np.power(u/50, -(1+alpha+beta*u))

        return Z_0 * F_w * (lmda**(n_w+1)) * (1-np.exp((-u*self.b*g_u_lmda)/(F_w * (lmda**n_w))))
    


    def loss_function(self, params):

        c1, c2, c3, c4 = params
        n0 = 5

        Z_0 = self.G_50 / (1 - np.exp(-50 * self.b))
        
        self.data['F_w'] = 1 + self.a1 * self.data['wickets_down'] + self.a2 * (self.data['wickets_down'] ** 2) + self.a3 * (self.data['wickets_down'] ** 3)
        self.data['n_w'] = n0 * self.data['F_w']

        lambda_val = self.data['match_factor']
        alpha = -1 / (1 + c1 * (lambda_val - 1) * np.exp(-c2 * (lambda_val - 1)))
        beta = -c3 * (lambda_val - 1) * np.exp(-c4 * (lambda_val - 1))

        u = self.data['balls_remaining'] / 6
        g_u_lambda = np.where(u != 0, np.power(u / 50, -(1 + alpha + beta * u)), 0)

        self.data['Z_dls'] = Z_0 * self.data['F_w'] * (lambda_val ** (self.data['n_w'] + 1)) * (
            1 - np.exp(-self.b * u * g_u_lambda / (self.data['F_w'] * (lambda_val ** self.data['n_w'])))
        )
        self.data['loss'] = (self.data['runs_remaining'] - self.data['predicted_value'])**2

        return self.data['loss'].mean()
    

    def optimize(self):
        
        initial_guess = [ 0.1, -0.1, 0.1, -0.1]
        print(f"{initial_guess = }")

        result = minimize(self.loss_function, initial_guess, method='Nelder-Mead')
        
        self.c1, self.c2, self.c3, self.c4 = result.x 
        
        print("Optimization Result:", result)




    


