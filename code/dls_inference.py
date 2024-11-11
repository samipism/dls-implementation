import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar
from main import Z_dls



def equation(x, params, team1_score):
    
    u, w = 50, 0  

    return Z_dls(params, u, w, x) - team1_score


def calculate_lambda(params, team1_score):

    solution = root_scalar(lambda x: equation(x, params, team1_score), bracket=[1,100])

    return solution.root


def inference(params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val):

    G_50, _, _, _, _, _, _, _, _ = params

    u_1 = 50 - team2_overs_down
    u_2 = 50 - (team2_overs_down + lost_overs_due_to_rain)

    P_u1_w = Z_dls(params, u_1, team2_wicktes_down, lambda_val)/Z_dls(params, 50, 0, lambda_val)
    P_u2_w = Z_dls(params, u_2, team2_wicktes_down, lambda_val)/Z_dls(params, 50, 0, lambda_val)

    R_1 = 1 * 100
    R_2 = (1 - P_u1_w + P_u2_w)*100

    team2_target = None

    if R_2 <= R_1:
        team2_target = team1_score * R_2/R_1
    else:
        team2_target = team1_score + G_50*(R_2 - R_1)

    return int(np.ceil(team2_target))


if __name__ == '__main__':

    params = [ 2.808e+02, -3.020e-02,  4.439e-01, -4.038e-01,  1.082e-01, 2.397e-01,  5.486e-01,  4.007e-01, -3.919e-01]

    team1_score = 320
    team2_wicktes_down = 4
    team2_overs_down = 10
    lost_overs_due_to_rain = 10

    lambda_val = calculate_lambda(params, team1_score)
    print(lambda_val)

    print(f"Team-1 Score: {team1_score}")

    print(f"Lambda value: {lambda_val}")

    print(f"Equation solver value: {equation(lambda_val, params, team1_score)}")

    print(f"Team 2's target: {inference(params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val)}")