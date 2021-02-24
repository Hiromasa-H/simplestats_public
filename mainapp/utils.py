# def handle_uploaded_file(f):
#     with open("main.csv","w") as destination:
#         destination.write(f)

        # data = preprocess(data,cat,ords,expvar,target)
        # mse_list = categorize(data,chosen_methods,chosen_metrics)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score,precision_score,recall_score, mean_squared_error,mean_absolute_error



def cat_ord(data,categoricals,ordinals):
    
    for category in categoricals:
        try:
            data[category] = data[category].astype("str")
            prefix = category
            data = pd.concat([data,pd.get_dummies(data[category],
                            prefix = prefix)],axis = 1).drop(category, axis = 1)
        except:
            pass

    for ordinal in ordinals:
        try:
            data[ordinal] = data[ordinal].astype("str")
        except:
            pass

    return data

def preprocess(data,categoricals,ordinals,explanatory_variables,target_variable):
    print(data.dtypes)

    X = data[explanatory_variables]
    y = data[target_variable]
    
    X = cat_ord(X,categoricals,ordinals)
    y = cat_ord(y,categoricals,ordinals)

    print(X.dtypes)
    print(y.dtypes)

    return X , y

def calc_metrics_c(y_true, y_pred,chosen_metrics):
    accuracy,precision,recall = "-","-","-"
    if "Accuracy" in chosen_metrics:
        accuracy = accuracy_score(y_true, y_pred).round(5)
    if "Precision" in chosen_metrics:
        precision = precision_score(y_true, y_pred).round(5)
    if "Recall" in chosen_metrics:
        recall = recall_score(y_true, y_pred).round(5)

    return accuracy,precision,recall

def categorize(X,y,chosen_methods,chosen_metrics,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    results = []
    #methods = ["LightGBM","GradientBoosting","RandomForest"]
    if "RandomForest" in chosen_methods:
        y_pred = RandomForest_(X_train,y_train,X_test)
        accuracy,precision,recall = calc_metrics_c(y_test,y_pred,chosen_metrics)
        result = {"method":"RandomForest","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)
    if "GradientBoosting" in chosen_methods:
        y_pred = GradientBoosting_(X_train,y_train,X_test)
        accuracy,precision,recall = calc_metrics_c(y_test,y_pred,chosen_metrics)
        result = {"method":"GradientBoosting","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)
    if "LightGBM" in chosen_methods:
        y_pred = LightGBM_(X_train,y_train,X_test,y_test)
        accuracy,precision,recall = calc_metrics_c(y_test,y_pred,chosen_metrics)
        result = {"method":"LightGBM","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)

    print(results)
    return results
    # mse_list = [
    #     {"method":"LGBM","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}},
    #     {"method":"RandomForest","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}}
    #     ]

def LightGBM_(X_train,y_train,X_test,y_test):
    import lightgbm as lgb
    hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
    }

    gbm = lgb.LGBMClassifier(**hyper_params)

    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=1000)

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    return y_pred

def GradientBoosting_(X_train,y_train,X_test):
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(n_estimators=500,
                                        random_state = 0)
    classifier.fit(X_train,y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred

    return y_pred

def RandomForest_(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=500,
                                        criterion = "entropy",
                                        random_state = 0)
    classifier.fit(X_train,y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred

    return y_pred

def calc_metrics_r(y_true, y_pred,chosen_metrics):
    MSE, RMSE, MAE = "-","-","-"
    if "MSE" in chosen_metrics:
        MSE = mean_squared_error(y_true,y_pred).round(5)
    if "RMSE" in chosen_metrics and "MSE" in chosen_metrics:
        RMSE = np.sqrt(MSE).round(5)
    if "MAE" in chosen_metrics:
        MAE = mean_absolute_error(y_true,y_pred).round(5)
    return MSE, RMSE, MAE

def regress(X,y,chosen_methods,chosen_metrics,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    results = []
    #"MultipleLinearRegression","PolynomialRegression","SupportVectorRegression"
    if "MultipleLinearRegression" in chosen_methods:
        y_pred = MultipleLinearRegression_(X_train,y_train,X_test)
        accuracy,precision,recall = calc_metrics_r(y_test,y_pred,chosen_metrics)
        result = {"method":"MultipleLinearRegression","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)
    if "PolynomialRegression" in chosen_methods:
        y_pred = PolynomialRegression_(X_train,y_train,X_test)
        accuracy,precision,recall = calc_metrics_r(y_test,y_pred,chosen_metrics)
        result = {"method":"PolynomialRegression","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)
    if "SupportVectorRegression" in chosen_methods:
        y_pred = SupportVectorRegression_(X_train,y_train,X_test)
        accuracy,precision,recall = calc_metrics_r(y_test,y_pred,chosen_metrics)
        result = {"method":"SupportVectorRegression","score":{
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall
        }}
        results.append(result)

    print(results)
    return results

#"MultipleLinearRegression","PolynomialRegression","SupportVectorRegression"
def MultipleLinearRegression_(X_train,y_train,X_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    #Predicting the test score results
    y_pred = regressor.predict(X_test)

    return y_pred

# def PolynomialRegression_(X_train,y_train,X_test):
#     return 

# def SupportVectorRegression_(X_train,y_train,X_test):
#     from sklearn.svm import SVR
#     regressor = SVR(kernel = "rbf")
#     regressor.fit(X_train,y_train)

#     sc_X = StandardScaler()
#     X_train = sc_X.fit_transform(X_train)

#     sc_y = StandardScaler()
#     y_train = sc_y.fit_transform(y_train.reshape(-1,1))

#     y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([X_test]))))
#     return y_pred