import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from statistics import mean, stdev
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

def main():
    
    filename = "ML_features_1min.csv"
    full_filename = os.path.join(ML_DIR, filename)
    data_df = pd.read_csv(full_filename, sep=' ')
    features = data_df.columns.tolist()
    
    # Normalize the data
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(data_df) # data - numpy.ndarray
    
    #data_df = pd.DataFrame(data, columns=features)  
    
    # Select summary statistics to display
    summary_stats = ['mean', 'std']
    
    # Compute summary statistics for each feature
    summary_stats_df1 = data_df.describe().T[summary_stats]
    summary_stats_df1 =  summary_stats_df1.reset_index()
    summary_stats_df1["feature"] = features
    summary_stats_df1 = summary_stats_df1[["feature", "mean", "std"]]


    filename = "ML_features_3min.csv"
    full_filename = os.path.join(ML_DIR, filename)
    data_df = pd.read_csv(full_filename, sep=' ')
    features = data_df.columns.tolist()
    
    # Normalize the data
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(data_df) # data - numpy.ndarray
    
    #data_df = pd.DataFrame(data, columns=features)  
    
    # Compute summary statistics for each feature
    summary_stats_df2 = data_df.describe().T[summary_stats]
    summary_stats_df2 =  summary_stats_df2.reset_index()
    summary_stats_df2["feature"] = features
    summary_stats_df2 = summary_stats_df2[["mean", "std"]]
    
    summary_stats_df = pd.concat([summary_stats_df1, summary_stats_df2], axis=1)
    
    #list_of_backslash = ['\\\\']*79
    #summary_stats_df["backslash"] = list_of_backslash    
    
    summary_stats_df.to_csv("stats.csv", sep = '&', header=True, float_format='%.6f', index=False)
    #summary_stats_df.to_csv("stats.csv", sep = '&', header=True, float_format='{:.3e}'.format, index=False)
     
main()

