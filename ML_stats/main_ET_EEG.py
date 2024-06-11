import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV#, train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit
from scipy.stats import randint
from sklearn import preprocessing
from scipy.stats import uniform

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = True
EQUAL_PERCENTILES = False

#MODEL = "LR"
#MODEL = "SVC"
#MODEL = "DT"
MODEL = "RF"
#MODEL = "HGBC"

selected_features = ['Left Pupil Diameter Mean', 'Left Blink Closing Amplitude Max',
                     'Left Blink Opening Amplitude Max',
                     'Right Blink Closing Amplitude Max', 'Head Pitch Max']

LABEL = "Workload"
#LABEL = "Vigilance"
#LABEL = "Stress"

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
        if LABEL == "Workload":
            th = eeg_series.quantile(.93)
        elif LABEL == "Vigilance":
            th = eeg_series.quantile(.1)
        else: #Stress
            th = eeg_series.quantile(.1)
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
    

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    
    
    data_df = data_df[selected_features]
    
    features_np = data_df.to_numpy()
    

    ###########################################################################
    #Shuffle data

    #print(features_np.shape)
    #print(scores_np.shape)
    
    if LABEL == "Workload":
        scores_np = scores_np[0,:] # WL
    elif LABEL == "Vigilance":
        scores_np = scores_np[1,:] # Vigilance
    else:
        scores_np = scores_np[2,:] # Stress

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    '''
    if BINARY:
        #Split into 2 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            th = eeg_series.quantile(.5)
        else:
            if LABEL == "Workload":
                th = eeg_series.quantile(.93)
            elif LABEL == "Vigilance":
                th = eeg_series.quantile(.1)
            else:
                th = eeg_series.quantile(.9)
        scores = [1 if score < th else 2 for score in scores]

    else:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            (th1, th2) = eeg_series.quantile([.33, .66])
        else:
            (th1, th2) = eeg_series.quantile([.52, .93])
            #(th1, th2) = eeg_series.quantile([.7, .48])
        scores = [1 if score < th1 else 3 if score > th2 else 2 for score in scores]

    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
    '''
    
    #print(type(features_np))
    features_np = np.array(features_np)
    
    # Spit the data into train and test
    '''
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, scores, test_size=0.1, shuffle=True
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
        
        clf = LogisticRegression(class_weight=weight_dict, solver='lbfgs')
        
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
        #WL, 3 classes: {'C': 3.918, 'penalty': l2} Acc=0.72, MacroF1=0.6 time:2sec
        #WL, binary: {'C': 3.779, 'penalty': l2} Acc=0.8, MacroF1=0.65    time:1sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 2.79, 'penalty': l2} Acc=0.7, MacroF1=0.56  time:2sec
        #WL, binary: {'C': 3.48, 'penalty': l2} Acc=0.8, MacroF1=0.65     time:1sec
        
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
        #WL, 3 classes: {'C': 6.748, 'kernel': poly, 'gamma': scale, 'degree': 5} Acc=0.74, MacroF1=0.61 time:6sec
        #WL, binary: {'C': 6.789, 'kernel': poly, 'gamma': scale, 'degree': 8} Acc=0.89, MacroF1=0.69    time:6sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'C': 9.167, 'kernel': rbf, 'gamma': scale, 'degree': 2} Acc=0.78, MacroF1=0.7   time:7sec
        #WL, binary: {'C': 3.154, 'kernel': poly, 'gamma': scale, 'degree': 6} Acc=0.87, MacroF1=0.69    time:5sec
        
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
        #WL, 3 classes: {'max_depth': 14} Acc=0.69, MacroF1=0.59 time:9sec
        #WL, binary: {'max_depth': 70} Acc=0.91, MacroF1=0.69    time:5sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 14} Acc=0.69, MacroF1=0.59 time:9sec
        #WL, binary: {'max_depth': 11} Acc=0.89, MacroF1=0.69    time:5sec
        
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
        '''
        param_grid = {'n_estimators': np.arange(100, 150, dtype=int),
             'max_depth': np.arange(1, 79, dtype=int),
             }
        search = GridSearchCV(clf, param_grid=param_grid, cv=10)
        '''
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
 
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
        #scoring = 'accuracy', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 12, 'n_estimators': 136} Acc=0.82 MacroF1=0.7   time:830sec
        #WL, binary: {'max_depth': 10, 'n_estimators': 261} Acc=0.95  MacroF1=0.77    time:530sec

        #scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 24, 'n_estimators': 237} Acc=0.83 MacroF1=0.72  time:835sec
        #WL, binary: {'max_depth': 7, 'n_estimators': 97} Acc=0.96  MacroF1=0.84      time:526sec
        
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
        #WL, 3 classes: {'max_depth': 12} Acc=0.86, MacroF1=0.77  time:101sec
        #WL, binary: {'max_depth': 18} Acc=0.95, MacroF1=0.77     time:106sec

        # scoring = 'f1_macro', n_iter=100, cv=5:
        #WL, 3 classes: {'max_depth': 12} Acc=0.86 MacroF1=0.77   time:253sec
        #WL, binary: {'max_depth': 4} Acc=0.94 MacroF1=0.82       time:131sec
    
    #importances = clf.feature_importances_
    #print(type(importances)) # class 'numpy.ndarray' 1x79
    
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
    