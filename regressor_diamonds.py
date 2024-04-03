from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_test_split_x_y(diamonds):
    X= diamonds.drop(["price"],axis =1)
    y= diamonds["price"]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test


def train_random_forest(diamonds):
    X_train,X_test,y_train,y_test = train_test_split_x_y(diamonds)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    return rf

def train_linear_regression(diamonds):
    X_train,X_test,y_train,y_test = train_test_split_x_y(diamonds)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr
