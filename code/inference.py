import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar

from main import Z_dls, lmda_equation, calculate_lambda, Z_std_u_w


def inference_std(Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain):

    G_50, _, _, _, _ = Z_std_params

    u_1 = 50 - team2_overs_down
    u_2 = 50 - (team2_overs_down + lost_overs_due_to_rain)

    P_u1_w = Z_std_u_w(Z_std_params, u_1, team2_wicktes_down)/Z_std_u_w(Z_std_params, 50, 0)
    P_u2_w = Z_std_u_w(Z_std_params, u_2, team2_wicktes_down)/Z_std_u_w(Z_std_params, 50, 0)

    R_1 = 1 * 100
    R_2 = (1 - P_u1_w + P_u2_w)*100

    team2_target = None

    if R_2 <= R_1:
        print(f"Condition satisfied: R_2 <= R_1")
        team2_target = team1_score * R_2/R_1
    else:
        print(f"Condition satisfied: R_2 > R_1")
        team2_target = team1_score + G_50*(R_2 - R_1)

    return int(np.ceil(team2_target))

def inference_dls(Z_dls_params, Z_std_params,  team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val):

    G_50, _, _, _, _ = Z_std_params

    u_1 = 50 - team2_overs_down
    u_2 = 50 - (team2_overs_down + lost_overs_due_to_rain)
    
    P_u1_w = Z_dls(Z_dls_params,Z_std_params, u_1, team2_wicktes_down, lambda_val)/Z_dls(Z_dls_params,Z_std_params, 50, 0, lambda_val)
    P_u2_w = Z_dls(Z_dls_params,Z_std_params, u_2, team2_wicktes_down, lambda_val)/Z_dls(Z_dls_params,Z_std_params, 50, 0, lambda_val)
    
    R_1 = 1 * 100
    R_2 = (1 - P_u1_w + P_u2_w)*100

    team2_target = None

    if R_2 <= R_1:
        team2_target = team1_score * R_2/R_1
    else:
        team2_target = team1_score + G_50*(R_2 - R_1)

    return int(np.ceil(team2_target))


def inference_ODI(Z_dls_params, Z_std_params,  team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain):

    G_50, _, _, _, _ = Z_std_params

    if team1_score <= G_50:
        return inference_std(Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain)
    else:
        lambda_val = calculate_lambda(Z_std_params, team1_score)
        print(f"Lambda value: {lambda_val}")
        print(f"Equation solver value: {lmda_equation(lambda_val, Z_std_params, team1_score)}")

        return inference_dls(Z_dls_params, Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val)




if __name__ == '__main__':

    Z_std_params = 2.808e+02, -3.020e-02,  4.439e-01, -4.038e-01,  1.082e-01
    Z_dls_params = -1.212e-01,  1.176e+00, -2.326e-02,  1.601e+00

    team1_score = 309
    team2_wicktes_down = 2
    team2_overs_down = 20
    lost_overs_due_to_rain = 30

    print(f"Team-1 Score: {team1_score}")

    print(f"Team 2's target: {inference_ODI(Z_dls_params, Z_std_params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain)}")
