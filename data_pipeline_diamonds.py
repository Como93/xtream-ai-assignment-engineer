import luigi
import requests
import pandas as pd
import csv
from data_preprocessing_diamonds import DataPrepocessing
import pickle
from regressor_diamonds import train_linear_regression, train_random_forest


CSV_FILE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"

class GetDiamondsFromApi(luigi.Task):
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/diamonds_from_github.csv")
    
    def run(self):
        download = requests.get(CSV_FILE)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        with self.output().open("w") as f:
            writer = csv.writer(f)
            writer.writerows(cr)
            f.close()
            
class GetDiamondsFromAssignment(luigi.Task):
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/diamonds_from_assignment.csv")
    
    def run(self):
        csv_from_assignment = pd.read_csv("./datasets/diamonds/diamonds.csv")
        csv_from_assignment.to_csv(self.output().path, index=False)

class DiamondsAggregation(luigi.Task):
    def requires(self):
        yield GetDiamondsFromApi()
        yield GetDiamondsFromAssignment()
    
    def run(self):
        diamonds_from_api = pd.read_csv(GetDiamondsFromApi().output().path)
        diamonds_from_assignment = pd.read_csv(GetDiamondsFromAssignment().output().path)
        diamonds_aggregated = pd.concat([diamonds_from_api,diamonds_from_assignment])
        diamonds_aggregated.to_csv(self.output().path)
    
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/diamonds_aggregated.csv")

class DiamondsDataPreparation(luigi.Task):
    def requires(self):
        return DiamondsAggregation()
    
    def run(self):
        dp = DataPrepocessing(pd.read_csv(DiamondsAggregation().output().path))
        diamonds_training = dp.preprocessing()
        diamonds_training.to_csv(self.output().path, index=False)
    
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/diamonds_training.csv")
    
class DiamondsTrainingPhaseLinearRegression(luigi.Task):
    def requires(self):
        return DiamondsDataPreparation()
    
    def run(self):
        linear_regression_model = train_linear_regression(pd.read_csv(DiamondsDataPreparation().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(linear_regression_model,f )
        
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/linear_model_regression.pkl") 

class DiamondsTrainingPhaseRandomForestRegression(luigi.Task):
    def requires(self):
        return DiamondsDataPreparation()
    
    def run(self):
        random_forest_model = train_linear_regression(pd.read_csv(DiamondsDataPreparation().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(random_forest_model,f )
        
    def output(self):
        return luigi.LocalTarget("./datasets/diamonds/random_forest_regression.pkl") 


if __name__ == '__main__':
    luigi.build([DiamondsTrainingPhaseLinearRegression(),DiamondsTrainingPhaseRandomForestRegression()])