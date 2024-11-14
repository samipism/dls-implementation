import pandas as pd
import matplotlib.pyplot as plt

from main import calculate_lambda, Z_dls, Z_std_u_w, load_data



def plot_observed_vs_predicted(data, Z_std_params, Z_dls_params):
    
    data = data[(data['season'] >= 2010) & (data['season'] <= 2015) & (data['innings'] == 2)]

    df = data.copy()

    df_300_plus = df[(df['total_runs'] >= 300)]
    grouped = df_300_plus.groupby(['match_id', 'innings'])['total_runs'].max()
    df_300_plus_avg = grouped.mean()

    lambda_val = calculate_lambda(Z_std_params, df_300_plus_avg)
    print(f"{lambda_val = }")

    x_val = []
    y_val_dl_observed = []
    y_val_dls_observed = []
    y_val_dl = []
    y_val_dls = []
    y_val_runrate = []
    

    for ball in range(0,301):

        y_val_runrate.append(ball/300)

        x_val.append(ball)

        temp_df = df[(df['balls_remaining'] == ball ) & (df['wickets_down'] == 0)]
        temp_df_300_plus = df_300_plus[(df_300_plus['balls_remaining'] == ball ) & (df_300_plus['wickets_down'] == 0)]

        df_runs_to_come = temp_df['runs_remaining']/temp_df['total_runs']
        df_300_plus_runs_to_come = temp_df_300_plus['runs_remaining']/temp_df_300_plus['total_runs']

        #print(f"{df_runs_to_come.shape = }, {df_300_plus_runs_to_come.shape = }")

        y_val_dl_observed.append(ball/300 if ball > 290 else df_runs_to_come.mean())
        y_val_dls_observed.append(ball/300 if ball > 285 else df_300_plus_runs_to_come.mean())

        y_val_dl.append(Z_std_u_w(Z_std_params, ball/6, 0)/Z_std_u_w(Z_std_params, 50, 0))
        y_val_dls.append(Z_dls(Z_dls_params,Z_std_params, ball/6, 0, lambda_val)/Z_dls(Z_dls_params, Z_std_params, 50, 0, lambda_val))

    
    plt.figure(figsize=(10, 6))
    plt.plot(x_val, y_val_dl_observed, color = 'red', linestyle='-', markersize = 2, marker = 's', linewidth=0.5,label='All ODI Matches [Avg 218]')
    plt.plot(x_val, y_val_dls_observed, color = 'blue', linestyle='-', markersize = 2, marker = 'o', linewidth = 0.5, label = 'High Scoring(300+) ODI Matches[Avg 330]')

    plt.plot(x_val, y_val_dl, color = 'red', label = 'Base DL(for score of 218)')
    plt.plot(x_val, y_val_dls, color = 'blue', label = 'DLS (for score of 330)')
    plt.plot(x_val, y_val_runrate, color = 'black', linestyle='--', label = 'Average Run Rate Line')
    plt.legend()
    plt.savefig('observed_vs_predicted.png',dpi=300)


def plot_resource_remaining(Z_std_params, Z_dls_params):

    z_std_resource_avail = [[] for _ in range(10)]
    z_dls_val_resource_avail = [[] for _ in range(10)]

    lambda_val = calculate_lambda(Z_std_params, 350)
    x_val = list(range(51))  

    for overs_used in range(51):
        for wickets_down in range(10):
            std_res =  Z_std_u_w(Z_std_params, 50 - overs_used, wickets_down) / Z_std_u_w(Z_std_params, 50, 0) 
            dls_res = Z_dls(Z_dls_params, Z_std_params,  50 - overs_used, wickets_down, lambda_val) / Z_dls(Z_dls_params, Z_std_params,  50, 0, lambda_val)
            z_std_resource_avail[wickets_down].append(std_res)
            z_dls_val_resource_avail[wickets_down].append(dls_res)

    plt.figure(figsize=(12, 8))
    for i in range(10):
        x,  = plt.plot(x_val, z_std_resource_avail[i],  linestyle='--', linewidth=1.5, color = 'red')
        y,  = plt.plot(x_val, z_dls_val_resource_avail[i],  linestyle='-', linewidth=1.5, color = 'blue')

    

    plt.xlabel('Overs used')
    plt.ylabel('Percentage of resources remaining')
    plt.title('Resource Availability by Wickets (Standard and DLS)')
    plt.legend([x, y], ['DL-STD', 'DLS'])
    plt.grid(True)
    plt.savefig('resource_remaining.png',dpi=300)

    # plt.figure(figsize=(10,6))
    # plt.plot(x_val, z_std_resource_avail[9])
    # plt.plot(x_val, z_dls_val_resource_avail[9])
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':

    path='../data/modified_ODI_data.csv'

    data = load_data(path)

    print(f"{data.shape = }")

    # over_data = data.groupby(['match_id','innings','over']).last().reset_index()
    # over_data['over'] = over_data['over'] + 1

    # print(f"{over_data.shape = }")

    Z_std_params = 1.995e+02 , 3.159e-02 , -1.196e-01,  -1.748e-03 , 3.840e-04

    Z_dls_params = 3.216e-01 , -4.633e+00,  3.518e-02 , -2.623e-01


    # Z_std_params = 2.045e+02, 3.159e-02, -1.196e-01, -1.748e-03,  3.840e-04
    # Z_dls_params = 3.124e-01, -5.117e+00, 4.113e-02, 5.560e-02

    plot_observed_vs_predicted(data, Z_std_params, Z_dls_params)

    plot_resource_remaining(Z_std_params, Z_dls_params)






    