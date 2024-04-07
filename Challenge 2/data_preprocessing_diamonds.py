import pandas as pd

class DataPrepocessing():
    def __init__(self,diamonds):
        self.diamonds = diamonds
    
    def preprocessing(self):
        diamondsReturn = self.diamonds
        diamondsReturn.drop_duplicates(inplace=True)
        diamondsReturn = self.drop_dimentionless_diamonds(diamondsReturn)
        diamondsReturn = self.drop_priceless_diamonds(diamondsReturn)
        diamondsReturn['cut'].replace(['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'],
                 [0,1,2,3,4], inplace=True)
        diamondsReturn['color'].replace(['G', 'E', 'F', 'H', 'D', 'I', 'J'],
                        [0,1,2,3,4,5,6], inplace=True)
        diamondsReturn['clarity'].replace(['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'IF', 'I1'],
                        [0,1,2,3,4,5,6,7], inplace=True)
        
        name_columns = ["carat","depth","table","x","y","z"]

        for column in name_columns:
            diamondsReturn = self.remove_outliers(diamondsReturn,column)

        return diamondsReturn
        
    def drop_dimentionless_diamonds(self,diamonds):
        diamonds = diamonds.drop(diamonds[diamonds["x"]==0].index)
        diamonds = diamonds.drop(diamonds[diamonds["y"]==0].index)
        diamonds = diamonds.drop(diamonds[diamonds["z"]==0].index)
        return diamonds
    
    def drop_priceless_diamonds(self,diamonds):
        diamonds = diamonds.drop(diamonds[diamonds["price"]<0].index)
        return diamonds
    
    def remove_outliers(self,diamonds,name_column):
        q1=diamonds[name_column].quantile(0.25)
        q3=diamonds[name_column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5*iqr
        upper_limit = q3 + 1.5*iqr
        diamonds_filtered = diamonds[(diamonds[name_column]>lower_limit) & (diamonds[name_column]<upper_limit)]
        return diamonds_filtered
