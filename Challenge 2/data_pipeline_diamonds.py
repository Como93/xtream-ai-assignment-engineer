import luigi
import requests
import pandas as pd
import csv
from data_preprocessing_diamonds import DataPrepocessing
import pickle
from regressor_diamonds import train_decision_tree


CSV_FILE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"

class GetDiamondsFromApi(luigi.Task):
    def run(self):
        filetmp = "./diamonds_from_github.csv"
        download = requests.get(CSV_FILE)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        with open(filetmp,"w") as f:
            writer = csv.writer(f)
            writer.writerows(cr)
            f.close()


    def output(self):
         return luigi.LocalTarget("./diamonds_from_github.csv")
            
class GetDiamondsFromAssignment(luigi.Task):
    def output(self):
        return luigi.LocalTarget("./diamonds_from_assignment.csv")
    
    def run(self):
        csv_from_assignment = pd.read_csv("../datasets/diamonds/diamonds.csv")
        csv_from_assignment.to_csv(self.output().path, index=False)

class DiamondsAggregation(luigi.Task):
    def requires(self):
        yield GetDiamondsFromApi()
        yield GetDiamondsFromAssignment()
    
    def run(self):
        diamonds_from_api = pd.read_csv(GetDiamondsFromApi().output().path)
        diamonds_from_assignment = pd.read_csv(GetDiamondsFromAssignment().output().path)
        diamonds_aggregated = pd.concat([diamonds_from_api,diamonds_from_assignment])
        diamonds_aggregated.to_csv(self.output().path, index=False)
    
    def output(self):
        return luigi.LocalTarget("./diamonds_aggregated.csv")

class DiamondsDataPreparation(luigi.Task):
    def requires(self):
        return DiamondsAggregation()
    
    def run(self):
        dp = DataPrepocessing(pd.read_csv(DiamondsAggregation().output().path))
        diamonds_training = dp.preprocessing()
        diamonds_training.to_csv(self.output().path, index=False)
    
    def output(self):
        return luigi.LocalTarget("./diamonds_training.csv")

class DiamondsTrainingPhase(luigi.Task):
    def requires(self):
        return DiamondsDataPreparation()
    
    def run(self):
        decision_tree_model = train_decision_tree(pd.read_csv(DiamondsDataPreparation().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(decision_tree_model,f )
        
    def output(self):
        return luigi.LocalTarget("./best_model_regression.pkl") 
