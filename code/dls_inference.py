import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar

# gg
from main import Z_dls




def equation(x, params, team1_score):

    G_50, b, a1, a2, a3 = params

    u, w, n0 = 50, 0, 5

    Z_0 = G_50/(1 - np.exp(-50*b))
    F_w = 1 + a1*w + a2 * (w**2) + a3 * (w**3)

    n_w = n0 * F_w

    return Z_0 * F_w * (x**(n_w+1)) * (1 - np.exp((-u*b)/(F_w * (x**n_w)))) - team1_score


def calculate_lambda(params, team1_score):

    solution = root_scalar(lambda x: equation(x, params, team1_score), bracket=[1,100])

    return solution.root


def inference(params, args,  team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val):

    G_50, _, _, _, _ = args

    u_1 = 50 - team2_overs_down
    u_2 = 50 - (team2_overs_down + lost_overs_due_to_rain)
    print("Her4e")
    print(Z_dls(params,args, 50, 0, lambda_val))

    P_u1_w = Z_dls(params,args, u_1, team2_wicktes_down, lambda_val)/Z_dls(params,args, 50, 0, lambda_val)
    P_u2_w = Z_dls(params,args, u_2, team2_wicktes_down, lambda_val)/Z_dls(params,args, 50, 0, lambda_val)
    print({f'{P_u1_w=}'})

    R_1 = 1 * 100
    R_2 = (1 - P_u1_w + P_u2_w)*100

    team2_target = None

    if R_2 <= R_1:
        team2_target = team1_score * R_2/R_1
    else:
        team2_target = team1_score + G_50*(R_2 - R_1)
    print(team2_target)

    return int(np.ceil(team2_target))


if __name__ == '__main__':

    args = 2.808e+02, -3.020e-02,  4.439e-01, -4.038e-01,  1.082e-01
    params = -1.212e-01,  1.176e+00, -2.326e-02,  1.601e+00

    team1_score = 309
    team2_wicktes_down = 2
    team2_overs_down = 20
    lost_overs_due_to_rain = 30

    lambda_val = calculate_lambda(args, team1_score)
    print(lambda_val)

    print(f"Team-1 Score: {team1_score}")

    print(f"Lambda value: {lambda_val}")

    print(f"Equation solver value: {equation(lambda_val, args, team1_score)}")

    print(f"Team 2's target: {inference(params, args, team1_score, team2_wicktes_down, team2_overs_down, lost_overs_due_to_rain, lambda_val)}")
