from main import Z_dls, calculate_lambda
import numpy as np
import scipy as sp
from scipy.optimize import minimize, root_scalar
import pandas as pd
import matplotlib.pyplot as plt

from main import load_data

def preprocess_data(data, params):
    # data = data[(data['target_score'] >290)]
    data = data[(data['season'] >= 2010) & (data['season'] <= 2015)]
    data = data[(data['total_runs'] > 300) | (data['target_score'] >300) ]
    data = data.copy()
    runs = np.where(data['innings'] == 1,data['total_runs'], data['target_score'] - 1)  
    lambdas = []
    for i in runs:
        lambdas.append(calculate_lambda(params, i))
    data['match_factor'] =lambdas
    # print(data)
    # print("max lambda", data["Match.Factor"].max())
    filtered_columns = ['innings','balls_remaining', 'wickets_down','runs_remaining', 'total_runs', 'match_factor']
    data = data[filtered_columns]
    print(data.shape)
    data.dropna(inplace=True, axis=0)
    return data


def calculate_loss1(params, args, data):
    total_loss = 0
    count = 0
    for ball in range(1, 301):
        for wicket in range(10):
            # Filter the data based on overs remaining and wickets down
            temp_data = data[(data['balls_remaining'] == ball) & (data['wickets_down'] == wicket)]
            # print(temp_data)

            if temp_data.shape[0] > 0:
                count += 1

                lambda_val = temp_data['match_factor'].to_numpy()
                predicted_value = Z_dls(params, args, ball/6, wicket, lambda_val)
                
                # print(f"{params=}, {args=}, {over=}, {wicket=}, {lambda_val=},{predicted_value=}")
                for i in predicted_value:
                    assert(i!=float('inf'))
                
                loss = (temp_data['runs_remaining'].values - predicted_value)**2

                total_loss += np.mean(loss)
        
    return total_loss/count


if __name__ == '__main__':

    input_path = '../data/modified_ODI_data.csv'
    G_50,b, a1, a2, a3 =  2.609e+02,  1.204e-02, -1.252e-01,  2.624e-03,  1.758e-04
    args = (G_50,b, a1, a2, a3)
    df_data = load_data(input_path)

    cleaned_data = preprocess_data(df_data, args)
    
    # print("Clean")
    # print(cleaned_data)

    # print(f"{cleaned_data.shape = }")

    #bounds = [(200, None), (1e-3, None),(1e-3, None),(1e-3, None),(1e-3, None)]

    initial_guess = [-0.1, 1, -0.1, 1]
    # args = 2.808e+02, -3.020e-02,  4.439e-01, -4.038e-01,  1.082e-01
    # initial_guess = [200,0.1,0.1,0.1,0.1]
    # bounds = [(0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10)]

    result = minimize(calculate_loss1, initial_guess, args = (args,cleaned_data), method='Nelder-Mead')
    print(f"{result = }")     













