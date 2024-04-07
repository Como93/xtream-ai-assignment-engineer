from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
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

def get_optimal_hyperparameter_random_forest(X_train_random_forest,y_train_random_forest):
    param_grid_random_forest = { 
        'n_estimators': [25, 50, 100, 150], 
        'max_depth': [3, 6, 9], 
        'max_leaf_nodes': [3, 6, 9], 
    } 

    grid_search_random_forest = GridSearchCV(RandomForestRegressor(), param_grid=param_grid_random_forest,cv=5) 
    grid_search_random_forest.fit(X_train_random_forest, y_train_random_forest) 
    return grid_search_random_forest.best_estimator_

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

    y_pred_decision_tree = dt.predict(X_test)

    mae_decision_tree, r2_score_decision_tree, mse_decision_tree, rmse_decision_tree = metrics(y_test, y_pred_decision_tree)
    decision_tree_output = ("Decision Tree Regression", r2_score_decision_tree, mae_decision_tree, mse_decision_tree, rmse_decision_tree)

    return dt, decision_tree_output


def train_random_forest(diamonds):
    X_train,X_test,y_train,y_test = train_test_split_x_y(diamonds)

    best_parameters_random_forest = get_optimal_hyperparameter_random_forest(X_train,y_train)

    rf = RandomForestRegressor(n_estimators = best_parameters_random_forest.n_estimators, 
                               max_depth = best_parameters_random_forest.max_depth,
                               max_leaf_nodes = best_parameters_random_forest.max_leaf_nodes )
    rf.fit(X_train, y_train)

    y_pred_random_forest = rf.predict(X_test)

    mae_random_forest, r2_score_random_forest, mse_random_forest, rmse_random_forest = metrics(y_test, y_pred_random_forest)
    random_forest_output = ("Random Forest Regression", r2_score_random_forest, mae_random_forest, mse_random_forest, rmse_random_forest)

    return rf, random_forest_output

def train_linear_regression(diamonds):
    X_train,X_test,y_train,y_test = train_test_split_x_y(diamonds)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_linear_regression = lr.predict(X_test)

    mae_linear_regression, r2_score_linear_regression, mse_linear_regression, rmse_linear_regression = metrics(y_test, y_pred_linear_regression)
    linear_regression_output = ("Linear Regression", r2_score_linear_regression, mae_linear_regression, mse_linear_regression, rmse_linear_regression)

    return lr, linear_regression_output
