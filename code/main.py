import numpy as np
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt



def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(data):
    
    data = data[(data['Runs.Remaining'] >= 0) & (data['Wickets.in.Hand'] > 0)]
    data = data.copy()

    data['Overs.Remaining'] = 50 - data['Over']
    data['Wickets.Down'] = 10 - data['Wickets.in.Hand']
    filtered_columns = ['Innings','Overs.Remaining', 'Wickets.Down','Runs.Remaining']
    data = data[filtered_columns]
    data.dropna(inplace=True, axis=0)

    # print(f"After preprocessing: {data.shape = }")

    # print(f"{data.columns = }")

    return data


def Z_std_u_w(params, u, w):
    G_50, b, a1, a2, a3 = params

    Z_0 = G_50/(1 - np.exp(-50*b))
    F_w = 1 + a1*w + a2 * (w**2) + a3 * (w**3)

    Z_std = Z_0 * F_w * (1 - np.exp((-u*b)/F_w))

    return Z_std 


# def calculate_loss(params, data):
#     loss = 0
#     for over in range(1,50):
#         for wicket in range(10):
#             temp_data = data[(data['Overs.Remaining'] == over) & (data['Wickets.Down'] == wicket)]
#             for k in range(temp_data.shape[0]):
#                 loss += (np.abs(temp_data['Runs.Remaining'].iloc[k] - Z_std_u_w(params, over, wicket)))

#     return loss/data.shape[0]

def calculate_loss(params, data):
    loss = 0
    count = 0
    for over in range(1, 50):
        for wicket in range(10):
            temp_data = data[(data['Overs.Remaining'] == over) & (data['Wickets.Down'] == wicket)]

            # print(f"{wicket = }, {over = }")
            # print(f"{temp_data['Runs.Remaining'].values} = ")
            #print(f"{Z_std_u_w(params, over, wicket) = }")


            if temp_data.shape[0] > 0:
                count += 1
                mean_value = np.mean(temp_data['Runs.Remaining'].values)
                #diff = np.abs(temp_data['Runs.Remaining'].values - Z_std_u_w(params, over, wicket))
                #loss += np.sum(diff)
                loss += (mean_value - Z_std_u_w(params, over, wicket))**2

    return loss / count


if __name__ == '__main__':

    input_path = '../data/04_cricket_1999to2011.csv'

    df_data = load_data(input_path)

    cleaned_data = preprocess_data(df_data)

    print(f"{cleaned_data.shape = }")

    # params = [ 2.500e+02, 1.000e-01 , 1.000e-01  ,1.000e-01 , 1.000e-01]

    # print(f"{calculate_loss(params, cleaned_data) = }")

    initial_guess = [200,0.1,0.1,0.1,0.1]

    #bounds = [(200, None), (1e-3, None),(1e-3, None),(1e-3, None),(1e-3, None)]

    result = minimize(calculate_loss, initial_guess, args = cleaned_data, method='L-BFGS-B')
    print(f"{result = }")     













