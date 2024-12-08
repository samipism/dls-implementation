
import matplotlib.pyplot as plt


class Graph: 

    def __init__(self, data_ob, dl_std_object, dl_pro_object, dl_stern_object):

        self.dl_std_ob = dl_std_object
        self.dl_pro_ob = dl_pro_object
        self.dl_stern_ob = dl_stern_object
        self.data = data_ob.odi_data.copy()
    

    def plot_observed_vs_predicted(self):
        
        new_data = self.data[(self.data['season'] >= 2010) & (self.data['season'] <= 2015) & (self.data['innings'] == 2)]

        df = new_data.copy()

        df_300_plus = df[(df['total_runs'] >= 300)]
        grouped = df_300_plus.groupby(['match_id', 'innings'])['total_runs'].max()
        df_300_plus_avg = grouped.mean()

        lambda_val = self.dl_pro_ob.calculate_lambda(df_300_plus_avg)

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
            temp_df_300_plus = df_300_plus[(df_300_plus['balls_remaining'] == ball) & (df_300_plus['wickets_down'] == 0)]

            df_runs_to_come = temp_df['runs_remaining']/temp_df['total_runs']
            df_300_plus_runs_to_come = temp_df_300_plus['runs_remaining']/temp_df_300_plus['total_runs']

            y_val_dl_observed.append(ball/300 if ball > 290 else df_runs_to_come.mean())
            y_val_dls_observed.append(ball/300 if ball > 285 else df_300_plus_runs_to_come.mean())

            y_val_dl.append(self.dl_std_ob.calculate_z(ball/6, 0)/self.dl_std_ob.calculate_z(50, 0))
            y_val_dls.append(self.dl_stern_ob.calculate_z(ball/6, 0, lambda_val)/self.dl_stern_ob.calculate_z(50, 0, lambda_val))
            
        plt.figure(figsize=(10, 6))
        plt.plot(x_val, y_val_dl_observed, color = 'red', linestyle='-', markersize = 2, marker = 's', linewidth=0.5,label='All ODI Matches [Avg 218]')
        plt.plot(x_val, y_val_dls_observed, color = 'blue', linestyle='-', markersize = 2, marker = 'o', linewidth = 0.5, label = 'High Scoring(300+) ODI Matches[Avg 330]')

        plt.plot(x_val, y_val_dl, color = 'red', label = 'Base DL(for score of 218)')
        plt.plot(x_val, y_val_dls, color = 'blue', label = 'DLS (for score of 330)')
        plt.plot(x_val, y_val_runrate, color = 'black', linestyle='--', label = 'Average Run Rate Line')

        plt.xlabel('Balls Remaining')
        plt.ylabel('Proportion of Runs To Come')
        plt.legend()
        plt.savefig('docs/observed_vs_predicted.png',dpi=300)


    def plot_resource_remaining(self):

        z_std_resource_avail = [[] for _ in range(10)]
        z_dls_val_resource_avail = [[] for _ in range(10)]

        # calculate a lambda for any team1 score as we need lambda in dls predictions
        lambda_val = self.dl_pro_ob.calculate_lambda(300)
        x_val = list(range(51))  

        for overs_used in range(51):
            for wickets_down in range(10):
                std_res =  self.dl_std_ob.calculate_z(50 - overs_used, wickets_down) / self.dl_std_ob.calculate_z(50, 0) 
                dls_res = self.dl_stern_ob.calculate_z(50 - overs_used, wickets_down, lambda_val) / self.dl_stern_ob.calculate_z(50, 0, lambda_val)
                z_std_resource_avail[wickets_down].append(std_res)
                z_dls_val_resource_avail[wickets_down].append(dls_res)

        plt.figure(figsize=(12, 8))
        for i in range(10):
            x,  = plt.plot(x_val, z_std_resource_avail[i],  linestyle='--', linewidth=1.5, color = 'red')
            y,  = plt.plot(x_val, z_dls_val_resource_avail[i],  linestyle='-', linewidth=1.5, color = 'blue')


        plt.xlabel('Overs used')
        plt.ylabel('Percentage of resources remaining')
        plt.title('Resource Availability by Wickets (Standard and DLS)')
        plt.legend([x, y], ['DL-STD', 'DL-Stern'])
        plt.grid(True)
        plt.savefig('docs/resource_remaining.png',dpi=300)
