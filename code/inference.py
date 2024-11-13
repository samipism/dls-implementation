import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar

from main import Z_std_u_w, calculate_lambda, lmda_equation


def inference(params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain):

    G_50, _, _, _, _ = params

    u_1 = 50 - team2_overs_down
    u_2 = 50 - (team2_overs_down + lost_overs_due_to_rain)

    P_u1_w = Z_std_u_w(params, u_1, team2_wicktes_down)/Z_std_u_w(params, 50, 0)
    P_u2_w = Z_std_u_w(params, u_2, team2_wicktes_down)/Z_std_u_w(params, 50, 0)

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


if __name__ == '__main__':

    params = 2.808e+02 , -3.020e-02,  4.440e-01, -4.038e-01,  1.082e-01

    team1_score = 287
    team2_wicktes_down = 3
    team2_overs_down = 17
    lost_overs_due_to_rain = 15

    lambda_val = calculate_lambda(params, team1_score)

    print(f"Team-1 Score: {team1_score}")

    print(f"Lambda value: {lambda_val}")

    print(f"Lambda Equation solver value: {lmda_equation(lambda_val, params, team1_score)}")

    print(f"Team 2's target: {inference(params, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain)}")


