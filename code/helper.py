
import pandas as pd
import numpy as np
from scipy.optimize import minimize, root_scalar

from main import Z_std_u_w, Z_dls, calculate_lambda


def calculate_loss(params, df):

    loss = 0

    for overs_left in range(50,-1,-1):
        
        for wicket_lost in range(10):

            pred = (Z_std_u_w(params, overs_left, wicket_lost) / Z_std_u_w(params, 50, 0)) * 100
            act = df.iloc[50-overs_left, wicket_lost + 1]

            #print(f"{overs_left = }, {wicket_lost = },{pred = }, {act = }")

            loss += (act - pred)**2
            

    return loss 

def make_file(Z_std_params, Z_dls_params):

    dl = [[] for i in range(10)]
    dls = [[] for i in range(10)]

    lambda_val = calculate_lambda(Z_std_params, 330)

    for overs_left in range(50,-1,-1):
        
        for wicket_lost in range(10):

            dl_val = (Z_std_u_w(Z_std_params, overs_left, wicket_lost) / Z_std_u_w(Z_std_params, 50, 0)) * 100
            #print(type(dl_val))

            dls_val = Z_dls(Z_dls_params,Z_std_params, overs_left, wicket_lost, lambda_val)/Z_dls(Z_dls_params,Z_std_params, 50, 0, lambda_val) * 100

            dl[wicket_lost].append(float(dl_val))
            dls[wicket_lost].append(float(dls_val))

    columns = ["Overs left"] + [f"Wicket lost = {i}" for i in range(1, 11)]



    df_dl = pd.DataFrame(dl)
    df_dls = pd.DataFrame(dls)

    df_dl = df_dl.T
    df_dls = df_dls.T

    df_dl.insert(0,'Overs_left',[50 - i for i in range(51)])
    df_dls.insert(0,'Overs_left', [50 - i for i in range(51)])
    
    df_dl.to_csv('dl_resource.csv', index=False, header=columns)
    df_dls.to_csv('dls_resource.csv', index=False, header=columns)

    

if __name__ == '__main__':

    # df = pd.read_csv('resource_data.txt', header=None, sep=' ')

    # initial_guess = [ 20.5 , 1.253e-02, -1.391e-01 , 5.942e-03 ,-2.945e-05]

    # result = minimize(calculate_loss, initial_guess, args = df, method='nelder-mead')

    # print(f"{result = }")  

    Z_std_params = 2.045e+02, 3.159e-02, -1.196e-01, -1.748e-03,  3.840e-04
    Z_dls_params = 3.124e-01, -5.117e+00, 4.113e-02 , 5.560e-02


    make_file(Z_std_params, Z_dls_params)   


    #calculate_loss(initial_guess, df)


    