import luigi
import requests
import pandas
import csv


CSV_FILE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"

class DiamondsTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget("diamonds_with_pipeline.csv")
    
    def run(self):
        download = requests.get(CSV_FILE)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        with self.output().open("w") as f:
            writer = csv.writer(f)
            writer.writerows(cr)
            f.close()

if __name__ == '__main__':
    luigi.build([DiamondsTask()],local_scheduler=True)