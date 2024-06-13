import warnings
warnings.filterwarnings('ignore')
import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from feature_engine.selection import DropCorrelatedFeatures

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = True
EQUAL_PERCENTILES = False

MODEL = "LR"
#MODEL = "SVC"
#MODEL = "DT"
#MODEL = "RF"
#MODEL = "HGBC"

N_ITER = 100
CV = 5
SCORING = 'f1_macro'
#SCORING = 'accuracy'

TIME_INTERVAL_DURATION = 60

np.random.seed(RANDOM_STATE)

def weight_classes(scores):
    
    vals_dict = {}
    for i in scores:
        if i in vals_dict.keys():
            vals_dict[i] += 1
        else:
            vals_dict[i] = 1
    total = sum(vals_dict.values())

    # Formula used:
    # weight = 1 - (no. of samples present / total no. of samples)
    # So more the samples, lower the weight

    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
    #print(weight_dict)
        
    return weight_dict

def getEEGThreshold(scores):
    #Split into 2 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        th = eeg_series.quantile(.5)
    else:
        th = eeg_series.quantile(.93)
    return th

def getEEGThresholds(scores):
    #Split into 3 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        (th1, th2) = eeg_series.quantile([.33, .66])
    else:
        (th1, th2) = eeg_series.quantile([.52, .93])
    return (th1, th2)


def main():
    
    filename = "ML_features_1min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df.drop('ATCO', axis=1)
    
    # Drop correlated features
    
    init_features = data_df.columns
    
    dcf = DropCorrelatedFeatures(threshold=0.999)
    data_df = dcf.fit_transform(data_df)
    
    new_features = data_df.columns
    
    features_dif = [item for item in init_features if item not in new_features]
    
    print(len(init_features))
    print(len(new_features))
    print(len(features_dif))
    
    print(features_dif)
    
    #print(new_features)
    
    features_np = data_df.to_numpy()

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    

    ###########################################################################
    #Shuffle data

    #print(features_np.shape)
    #print(scores_np.shape)
    
    scores_np = scores_np[0,:] # Workload

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    
    # Spit the data into train and test
    
    data_split = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    
    for i, (train_idx, test_idx) in enumerate(data_split.split(features_np, scores)):
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
    
    
    #normalize train set
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
       
    if  BINARY:
        th = getEEGThreshold(y_train)
        y_train = [1 if score < th else 2 for score in y_train]
    else:
        (th1, th2) = getEEGThresholds(y_train)
        y_train = [1 if score < th1 else 3 if score > th2 else 2 for score in y_train]
    
    print("EEG")
    number_of_classes = len(set(y_train))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(y_train)
    
    #normalize test set
    X_test = scaler.transform(X_test)
    
    if  BINARY:
        y_test = [1 if score < th else 2 for score in y_test]
    else:
        y_test = [1 if score < th1 else 3 if score > th2 else 2 for score in y_test]
        
    ################################# Fit #####################################
    
    print(f"Model: {MODEL}")
    print(f"Scoring: {SCORING}, n_iter: {N_ITER}, cv: {CV}")

    if MODEL == "LR":
        
        clf = LogisticRegression(class_weight=weight_dict, solver='lbfgs',
                                 max_iter=1000)
        
        param_dist = {
            'C': uniform(loc=0, scale=4),  # Regularization parameter
            'penalty': ['l1', 'l2'],       # Penalty norm
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif MODEL == "SVC":
        
        clf = SVC(class_weight=weight_dict)
        
        param_dist = {
            'C': uniform(loc=0, scale=10),  # Regularization parameter
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            'degree': randint(1, 10)  # Degree of polynomial kernel
            }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif  MODEL == "DT":
        
        clf = DecisionTreeClassifier(class_weight=weight_dict)

        # Use random search to find the best hyperparameters
        param_dist = {
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif  MODEL == "RF":
        clf = RandomForestClassifier(class_weight=weight_dict,
                                     #bootstrap=False,
                                     max_features=None,
                                     random_state=RANDOM_STATE)
        
        # Use random search to find the best hyperparameters
        param_dist = {'n_estimators': randint(50,500),
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER, 
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)

        # Fit the search object to the data
        search.fit(X_train, y_train)
 
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
                
    elif  MODEL == "HGBC":
        clf = HistGradientBoostingClassifier(class_weight='balanced',
                                             random_state=RANDOM_STATE)

        # Use random search to find the best hyperparameters
        param_dist = {
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
    
    #importances = clf.feature_importances_
    #print(type(importances)) # class 'numpy.ndarray' 1x79
    
    ############################## Predict ####################################
    
    y_pred = best_clf.predict(X_test)

    print("Shape at output after classification:", y_pred.shape)

    ############################ Evaluate #####################################
    
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
    print("Accuracy:", accuracy)
    print("Macro F1-score:", f1_macro)
    
    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    