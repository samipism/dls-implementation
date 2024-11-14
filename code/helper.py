
import pandas as pd
import numpy as np
from scipy.optimize import minimize, root_scalar


def Z_std_u_w(params, u, w):
    G_50, b, a1, a2, a3 = params

    Z_0 = G_50/(1 - np.exp(-50*b))
    F_w = 1 + a1*w + a2 * (w**2) + a3 * (w**3)

    Z_std = Z_0 * F_w * (1 - np.exp((-u*b)/F_w))

    return Z_std 


def calculate_loss(params, df):

    loss = 0

    for overs_left in range(50,-1,-1):
        
        for wicket_lost in range(10):

            pred = (Z_std_u_w(params, overs_left, wicket_lost) / Z_std_u_w(params, 50, 0)) * 100
            act = df.iloc[50-overs_left, wicket_lost + 1]

            #print(f"{overs_left = }, {wicket_lost = },{pred = }, {act = }")

            loss += (act - pred)**2
            

    return loss 


if __name__ == '__main__':

    df = pd.read_csv('resource_data.txt', header=None, sep=' ')

    initial_guess = [ 20.5 , 1.253e-02, -1.391e-01 , 5.942e-03 ,-2.945e-05]

    result = minimize(calculate_loss, initial_guess, args = df, method='nelder-mead')

    # bounds = [(1e2, 1e3), (1e-5, 1e2), (-1e1, 1e1), (-1e1, 1e1), (-1e1, 1e1)]
    # result = minimize(calculate_loss, initial_guess, args=(df,), method='L-BFGS-B', bounds=bounds)

    print(f"{result = }")     


    #calculate_loss(initial_guess, df)


    