import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from scipy.stats import uniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from sklearn.base import clone
from sklearn.inspection import permutation_importance


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


RANDOM_STATE = 0

BINARY = True

PLOT = True

if BINARY == True:
    MODEL = "SVC"
else:
    MODEL = "HGBC"

N_ITER = 100
CV = 5
SCORING = 'f1_macro'
#SCORING = 'accuracy'

TIME_INTERVAL_DURATION = 180

np.random.seed(RANDOM_STATE)

# Function to calculate permutation importance using sklearn
def calculate_permutation_importance(model, X_val, y_val, scoring=SCORING, n_repeats=5):
    result = permutation_importance(model, X_val, y_val, scoring=scoring,
                                    n_repeats=n_repeats, random_state=RANDOM_STATE)
    return result.importances_mean

# Custom RFE class with permutation importance
class RFEPermutationImportance:
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
        
        self.estimator_.fit(X[features], y)
        
        test_accuracy = accuracy_score(y_test, self.estimator_.predict(X_test[features]))
        self.accuracies_.append((len(features), test_accuracy))
        
        test_f1_score = f1_score(y_test, self.estimator_.predict(X_test[features]), average='macro')
        self.f1_scores_.append((len(features), test_f1_score))
        
        while len(features) > self.min_features_to_select:
            self.estimator_.fit(X[features], y)
            importance = calculate_permutation_importance(self.estimator_, X[features], y, n_repeats=self.n_repeats)
            
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


def main():
    
    filename = "ML_features_3min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df.drop('ATCO', axis=1)
    

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    
    features_np = data_df.to_numpy()

    ###########################################################################
    #Shuffle data

    #print(features_np.shape)
    #print(scores_np.shape)
    
    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)
    
    scores = list(scores_np)
    
    data_df = pd.DataFrame(features_np, columns=data_df.columns)
    
    
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
    
    test_accuracies = []
    acc_num_features_list = []
    test_f1_scores = []
    f1_num_features_list = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data_df, scores, test_size=0.1,
                                                        random_state=RANDOM_STATE,
                                                        shuffle=True)

    #normalize train set
    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=data_df.columns)
    
    #normalize test set
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=data_df.columns)
    
    
    ################################# Fit #####################################
    
    print(f"Model: {MODEL}")
    print(f"Scoring: {SCORING}, n_iter: {N_ITER}, cv: {CV}")
        
    if MODEL == "SVC":

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
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_
 
    # Perform RFE with Permutation Importance
    rfe = RFEPermutationImportance(best_clf, min_features_to_select=58, n_repeats=5)

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
    print(selected_features)
               
    X_train_selected = X_train[selected_features]

    # Final model training with selected features
    best_clf.fit(X_train_selected, y_train)
    

    ############################## Predict ####################################

    X_test_selected = X_test[selected_features]
    y_pred = best_clf.predict(X_test_selected)

    print("Shape at output after classification:", y_pred.shape)
    
    ############################ Evaluate #####################################
    
    #print(y_test)
    #print(y_pred)
        
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
    print("Accuracy:", accuracy)
    print("Macro F1-score:", f1_macro)

    #print(test_accuracies)
    #print(test_f1_scores)
    
    if PLOT:
    
        filename = "acc_scoring_ch_"
        if BINARY:
            filename = filename + "binary_"
        else:
            filename = filename + "3classes_"
        filename = filename + MODEL

        # Plot accuracies
        fig1, ax1 = plt.subplots()
        ax1.plot(acc_num_features_list, test_accuracies, marker='o')
        ax1.set_xlabel('Number of Features', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.tick_params(axis='both', which='minor', labelsize=10)
        plt.grid(True)
        acc_filename = filename + "_acc.png"
        full_filename = os.path.join(FIG_DIR, acc_filename)
        plt.savefig(full_filename, dpi=600)
        plt.show()
        
        plt.clf()
        
        # Plot F1-scores
        fig2, ax2 = plt.subplots()
        ax2.plot(f1_num_features_list, test_f1_scores, marker='o')
        ax2.set_xlabel('Number of Features', fontsize=14)
        ax2.set_ylabel('F1-score', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='minor', labelsize=10)
        plt.grid(True)
        f1_filename = filename + "_f1.png"
        full_filename = os.path.join(FIG_DIR, f1_filename)
        plt.savefig(full_filename, dpi=600)
        plt.show()


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
