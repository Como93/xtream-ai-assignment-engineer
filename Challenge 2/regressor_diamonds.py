from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return mae, r2, mse, rmse

def train_test_split_x_y(diamonds):
    X= diamonds.drop(["price"],axis =1)
    y= diamonds["price"]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test

def get_optimal_hyperparameter_decision_tree(X_train_decision_tree,y_train_decision_tree):
    param_grid_decision_tree = {
          'max_depth':[3,5,7,10,15],
          'min_samples_leaf':[3,5,10,15,20],
          'min_samples_split':[8,10,12,18,20,16]
    }

    grid_search_decision_tree = GridSearchCV(DecisionTreeRegressor(), 
                           param_grid=param_grid_decision_tree,cv=5) 
    grid_search_decision_tree.fit(X_train_decision_tree, y_train_decision_tree) 
    return grid_search_decision_tree.best_estimator_


def train_decision_tree(diamonds):
    X_train,X_test,y_train,y_test = train_test_split_x_y(diamonds)

    best_parameters_decision_tree = get_optimal_hyperparameter_decision_tree(X_train,y_train)

    dt = DecisionTreeRegressor(max_depth = best_parameters_decision_tree.max_depth,
                               min_samples_leaf = best_parameters_decision_tree.min_samples_leaf,
                               min_samples_split = best_parameters_decision_tree.min_samples_split)
    dt.fit(X_train, y_train)

    return dt

