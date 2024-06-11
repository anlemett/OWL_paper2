import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.model_selection import RandomizedSearchCV #, train_test_split
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import  KFold
from sklearn import preprocessing

from feature_engine.selection import DropCorrelatedFeatures

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = False

#MODEL = "LR"
#MODEL = "SVC"
#MODEL = "DT"
#MODEL = "RF"
MODEL = "HGBC"

N_ITER = 100
CV = 5
SCORING = 'f1_macro'
#SCORING = 'accuracy'

TIME_INTERVAL_DURATION = 180

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


def main():
    
    filename = "ML_features_3min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df.drop('ATCO', axis=1)
    
    # Drop correlated features
    
    dcf = DropCorrelatedFeatures(threshold=0.95)
    data_df = dcf.fit_transform(data_df)
    
    print(len(data_df.columns))
    print(data_df.columns)
    
    features_np = data_df.to_numpy()

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    ###########################################################################
    #Shuffle data

    #print(features_np.shape)
    #print(scores_np.shape)

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    
    if BINARY:
        scores = [1 if score < 4 else 2 for score in scores]
    else:
        scores = [1 if score < 2 else 3 if score > 3 else 2 for score in scores]

    #print(scores)
    
    print("CHS")
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
        
    # Spit the data into train and test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        TS_np, scores, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
        )
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(
        f"Length of train  X : {len(X_train)}\nLength of test X : {len(X_test)}\nLength of train Y : {len(y_train)}\nLength of test Y : {len(y_test)}"
        )
    '''

    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)

    for i, (train_idx, test_idx) in enumerate(rs.split(features_np)):
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]

    #normalize train set
    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    
    #normalize test set
    X_test = scaler.transform(X_test)
    
    
    ################################# Fit #####################################
    
    print(f"Model: {MODEL}")
    print(f"Scoring: {SCORING}, n_iter: {N_ITER}, cv: {CV}")

    if MODEL == "LR":

        clf = LogisticRegression(class_weight=weight_dict, solver='liblinear')
        
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
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        # scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 1.657, 'penalty': l2} Acc=0.69, MacroF1=0.55 time:1sec
        #WL, binary: {'C': 0.019, 'penalty': l2} Acc=0.85, MacroF1=0.46    time:1sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 3.779, 'penalty': l2} Acc=0.72, MacroF1=0.57 time:1sec
        #WL, binary: {'C': 0.077, 'penalty': l2} Acc=0.85, MacroF1=0.65    time:1sec
        
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
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        # scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 6.748, 'kernel': poly, 'gamma': scale, 'degree': 5} Acc=0.81, MacroF1=0.72 time:1sec
        #WL, binary: {'C': 0.567, 'kernel': sigmoid, 'gamma': scale, 'degree': 2} Acc=0.9, MacroF1=0.47  time:1sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 6.748, 'kernel': poly, 'gamma': scale, 'degree': 5} Acc=0.81, MacroF1=0.72 time:1sec
        #WL, binary: {'C': 3.154, 'kernel': poly, 'gamma': scale, 'degree': 6} Acc=0.93, MacroF1=0.81    time:1sec

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
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        # scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 5} Acc=0.7, MacroF1=0.65   time:2sec
        #WL, binary: {'max_depth': 37} Acc=0.91, MacroF1=0.73    time:1sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 33} Acc=0.69, MacroF1=0.64 time:9sec
        #WL, binary: {'max_depth': 59} Acc=0.91, MacroF1=0.73    time:1sec

    
    elif  MODEL == "RF":
        
        clf = RandomForestClassifier(class_weight=weight_dict,
                                     bootstrap=False,
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
        '''
        param_grid = {'n_estimators': np.arange(100, 150, dtype=int),
             'max_depth': np.arange(1, 79, dtype=int),
             }
        search = GridSearchCV(clf, param_grid=param_grid, cv=CV)
        '''
        # Fit the search object to the data
        search.fit(X_train, y_train)

        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        # scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': , 'n_estimators': } Acc=0., MacroF1=0. time:sec
        #WL, binary: {'max_depth': 45, 'n_estimators': 97} Acc=0.91, MacroF1=0.73   time:488sec
        
        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 5, 'n_estimators': 287} Acc=0.7, MacroF1=0.65 time:864sec
        #WL, binary: {'max_depth': 45, 'n_estimators': 97} Acc=0.91, MacroF1=0.73   time:501sec
        
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
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_
 
        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
        
        # scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 11} Acc=0.82, MacroF1=0.7  time:216sec
        #WL, binary: {'max_depth': 4} Acc=0.91, MacroF1=0.73     time:45sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 45} Acc=0.84, MacroF1=0.72 time:204sec
        #WL, binary: {'max_depth': 4} Acc=0.91, MacroF1=0.73     time:40sec
    
    ############################## Predict ####################################

    y_pred = best_clf.predict(X_test)

    print("Shape at output after classification:", y_pred.shape)
    
    ############################ Evaluate #####################################
    
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    if BINARY:
        precision = precision_score(y_pred=y_pred, y_true=y_test, average='binary')
        recall = recall_score(y_pred=y_pred, y_true=y_test, average='binary')
        f1 = f1_score(y_pred=y_pred, y_true=y_test, average='binary')
    else:
        recall = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        f1 = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        
    f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
    print("Accuracy:", accuracy)
    #print("Precision: ", precision)
    #print("Recall: ", recall)
    #print("F1-score:", f1)
    print("Macro F1-score:", f1_macro)

    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
