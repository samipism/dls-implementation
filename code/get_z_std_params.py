import numpy as np
from scipy.optimize import minimize, root_scalar
import pandas as pd
import matplotlib.pyplot as plt

from main import load_data, Z_std_u_w, Z_pro_u_w, lmda_equation, calculate_lambda


def preprocess_data(data):
    data = data.copy()
    data = data[(data['season'] >= 2000) & (data['season'] <= 2012)]
    filtered_columns = ['innings','balls_remaining', 'wickets_down','runs_remaining']
    data = data[filtered_columns]
    data.dropna(inplace=True, axis=0)
    return data


def calculate_loss(params, data):
    total_loss = 0
    count = 0
    for ball in range(1, 301):
        for wicket in range(10):
            temp_data = data[(data['balls_remaining'] == ball) & (data['wickets_down'] == wicket)]
            z_val = Z_std_u_w(params, ball/6, wicket)

            if temp_data.shape[0] > 0:
                count += 1
                loss = (temp_data['runs_remaining'].values - z_val)**2
                total_loss += np.mean(loss)

    return total_loss / count

if __name__ == '__main__':

    input_path = '../data/modified_ODI_data.csv'

    df_data = load_data(input_path)
    cleaned_data = preprocess_data(df_data)
    print(f"{cleaned_data.shape = }")

    #bounds = [(200, None), (1e-3, None),(1e-3, None),(1e-3, None),(1e-3, None)]
    initial_guess = [200,.1,.1,.1,.1]

    result = minimize(calculate_loss, initial_guess, args = cleaned_data, method='TNC')
    print(f"{result = }")     

