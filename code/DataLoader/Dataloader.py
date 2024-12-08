import pandas as pd 

class Dataloader: 

    def __init__(self, path_t20i = '../data/modified_T20_data.csv', path_odi='../data/modified_ODI_data.csv'): 

        self.t20i_data = self.get_data(path_t20i)
        self.odi_data = self.get_data(path_odi)


    def get_data(self, path):

        data = pd.read_csv(path)

        filtered_columns = ['match_id','season', 'innings', 'balls_remaining', 'wickets_down','runs_remaining', 'total_runs', 'target_score']

        data = data[filtered_columns]

        data.dropna(inplace=True, axis=0)

        return data


if __name__ == '__main__':

    data_ob = Dataloader()

    print(f"{data_ob.t20i_data.shape = }, {data_ob.odi_data.shape = }")



    

    





