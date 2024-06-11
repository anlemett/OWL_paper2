import warnings
warnings.filterwarnings('ignore')
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = True
EQUAL_PERCENTILES = False

MODEL = "RF"

LABEL = "Workload"
#LABEL = "Vigilance"
#LABEL = "Stress"

N_ITER = 100
CV = 5
SCORING = 'f1_macro'
#SCORING = 'accuracy'

TIME_INTERVAL_DURATION = 60

np.random.seed(RANDOM_STATE)

# Custom RFE class with embedded importance
class RFEEmbeddedImportance:
    def __init__(self, estimator, n_features_to_select=None, step=1, min_features_to_select=1, n_repeats=5):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.n_repeats = n_repeats
        self.accuracies_ = []
        self.f1_scores_ = []

    def fit(self, X, y, X_test, y_test):
        self.estimator_ = clone(self.estimator)
        features = list(X.columns)
        while len(features) > self.min_features_to_select:
            self.estimator_.fit(X[features], y)
            importance = self.estimator_.feature_importances_
            
            # Identify least important feature
            least_important_feature_index = np.argmin(importance)
            least_important_feature = features[least_important_feature_index]
            
            # Remove the least important feature
            features.remove(least_important_feature)
            
            print(f'Removed feature: {least_important_feature}')
            print(f'Remaining features: {len(features)}')
            
            # Evaluate and store test accuracy and F1-score without removed feature
            
            self.estimator_.fit(X[features], y)
            
            test_accuracy = accuracy_score(y_test, self.estimator_.predict(X_test[features]))
            self.accuracies_.append((len(features), test_accuracy))
            
            test_f1_score = f1_score(y_test, self.estimator_.predict(X_test[features]), average='macro')
            self.f1_scores_.append((len(features), test_f1_score))
        
        self.support_ = np.isin(X.columns, features)
        self.ranking_ = np.ones(len(X.columns), dtype=int)
        self.ranking_[~self.support_] = len(X.columns) - np.sum(self.support_) + 1
        
        return self

    def transform(self, X):
        return X.loc[:, self.support_]


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
    
    data_df = data_df.drop('ATCO', axis=1)
    
    print(len(data_df.columns))
    #print(data_df.columns)
    

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    

    noncorr_features = ['Saccades Number', 'Saccades Duration Mean', 'Saccades Duration Std',
       'Saccades Duration Median', 'Saccades Duration Max',
       'Fixation Duration Mean', 'Fixation Duration Median',
       'Fixation Duration Max', 'Left Pupil Diameter Mean',
       'Right Pupil Diameter Mean', 'Left Blink Closing Amplitude Mean',
       'Head Heading Mean', 'Head Pitch Mean', 'Head Roll Mean',
       'Left Pupil Diameter Std', 'Right Pupil Diameter Std',
       'Head Heading Std', 'Head Pitch Std', 'Head Roll Std',
       'Left Pupil Diameter Min', 'Right Pupil Diameter Min',
       'Head Heading Min', 'Head Pitch Min', 'Head Roll Min',
       'Left Pupil Diameter Max', 'Right Pupil Diameter Max',
       'Left Blink Closing Amplitude Max', 'Left Blink Closing Speed Max',
       'Left Blink Opening Speed Max', 'Right Blink Closing Amplitude Max',
       'Right Blink Closing Speed Max', 'Head Heading Max', 'Head Pitch Max',
       'Head Roll Max', 'Head Heading Median']

    data_df = data_df[noncorr_features]
    
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
    
    #data_df = pd.DataFrame(features_np, columns=data_df.columns)
    data_df = pd.DataFrame(features_np, columns=noncorr_features)
    
    
    # Spit the data into train and test
    
    test_accuracies = []
    acc_num_features_list = []
    test_f1_scores = []
    f1_num_features_list = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data_df, scores, test_size=0.1,
                                                        random_state=RANDOM_STATE,
                                                        shuffle=True)
    
    #data_split = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    #data_split = KFold(n_splits=10, random_state=None, shuffle=False)
    '''
    data_split = KFold(n_splits=10, random_state=RANDOM_STATE, shuffle=True)
    
    for i, (train_idx, test_idx) in enumerate(data_split.split(features_np, scores)):
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
    '''
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    f1_scores = []

    for train_index, test_index in kf.split(data_df):
        X_train, X_test = data_df.iloc[train_index], data_df.iloc[test_index]
        y_train, y_test = [scores[i] for i in train_index], [scores[i] for i in test_index]
    '''
    #normalize train set
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=data_df.columns)
    
      
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
    X_test = pd.DataFrame(X_test, columns=data_df.columns)
    
    if  BINARY:
        y_test = [1 if score < th else 2 for score in y_test]
    else:
        y_test = [1 if score < th1 else 3 if score > th2 else 2 for score in y_test]
        
    ################################ Fit #####################################
        
    print(f"Model: {MODEL}")
    print(f"Scoring: {SCORING}, n_iter: {N_ITER}, cv: {CV}")
    
    
    if  MODEL == "RF":
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
        
        
    # Perform RFE with Embedded Importance
    rfe = RFEEmbeddedImportance(best_clf, min_features_to_select=1)

    rfe.fit(X_train, y_train, X_test, y_test)
    
    # Store accuracies for plotting
    for num_features, accuracy in rfe.accuracies_:
        acc_num_features_list.append(num_features)
        test_accuracies.append(accuracy)
    
    # Store accuracies for plotting
    for num_features, train_f1_score in rfe.f1_scores_:
         f1_num_features_list.append(num_features)
         test_f1_scores.append(train_f1_score)
    
    # Select the remaining features
    selected_features = list(X_train.columns[rfe.support_])
               
    X_train_selected = X_train[selected_features]

    # Final model training with selected features
    clf.fit(X_train_selected, y_train)
        
    
    ############################## Predict ####################################
    
    X_test_selected = X_test[selected_features]
    y_pred = clf.predict(X_test_selected)
        
    print("Shape at output after classification:", y_pred.shape)

    ############################ Evaluate #####################################
        
    print(selected_features)
    
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
    
    
    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(acc_num_features_list, test_accuracies, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title(MODEL)
    plt.grid(True)
    plt.show()
    
    print(test_accuracies)
    
    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(f1_num_features_list, test_f1_scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('F1-score')
    plt.title(MODEL)
    plt.grid(True)
    plt.show()
    
    print(test_f1_scores)
    
    print(f1_num_features_list)
        
    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    