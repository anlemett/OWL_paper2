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
FIG_DIR = os.path.join(".", "Figures")

#TIME_INTERVAL_DURATION = 60
TIME_INTERVAL_DURATION = 180

def main():
    
    if TIME_INTERVAL_DURATION == 60:
        filename = "ML_features_1min.csv"
    else:
        filename = "ML_features_3min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    #temp_df = data_df[["Fixation Duration Min"]]
    #temp_df.to_csv("temp.csv", sep = ' ', header=True, float_format='%.3f', index=False)

    temp_df = data_df[["Left Pupil Diameter min", 
                       "Left Pupil Diameter max", 
                       "Left Blink Opening Amplitude min",
                       ]]
    temp_df.to_csv("temp.csv", sep = ' ', header=True, float_format='%.3f', index=False)
    
    features = data_df.columns
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data_df) # data - numpy.ndarray
    
    data_df = pd.DataFrame(data, columns=features)  
    
    # Select summary statistics to display
    summary_stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    # Compute summary statistics for each feature
    summary_stats_df = data_df.describe().T[summary_stats]
      
    summary_stats_df =  summary_stats_df.reset_index()

    # Create parallel coordinates plot
    plt.figure(figsize=(10, 6))
    pd.plotting.parallel_coordinates(summary_stats_df, class_column='index')
    plt.xticks(range(7), summary_stats)
    plt.xlabel('Summary Statistic')
    plt.ylabel('Normalized Value')
    #plt.title('Parallel Coordinates Plot of 79 Features Summary Statistics')
    
    plt.gca().legend_.remove()
    if TIME_INTERVAL_DURATION == 60:
        filename = "Features_1min.png"
    else:
        filename = "Features_3min.png"
    full_filename = os.path.join(FIG_DIR, filename)
    plt.savefig(full_filename)
    #plt.show()
    plt.clf()
    
    
    feature_data = data_df["Fixation Duration Min"]
    plt.hist(feature_data, bins=30, edgecolor="black")  # Adjust the number of bins as needed
    plt.xlabel("Saccades Duration Min")
    plt.ylabel("Number of values")
    plt.grid(True)
    plt.show()
    plt.clf()
    
    means = []
    sds = []

    for feature in features:
        feature_data = data_df[feature]
        # Plot the distribution as a histogram
        plt.hist(feature_data, bins=30, edgecolor="black")  # Adjust the number of bins as needed
        plt.xlabel(feature)
        plt.ylabel("Number of values")
        plt.grid(True)
        full_filename = os.path.join(FIG_DIR, feature + ".png")
        plt.savefig(full_filename)
        plt.clf()
        means.append(mean(feature_data))
        sds.append(stdev(feature_data))
        
    stats_df = pd.DataFrame()
    stats_df["feature"] = features
    stats_df["mean"] = means
    stats_df["sd"] = sds
    
    stats_df.to_csv("stats.csv", sep = '&', header=True, float_format='%.3f', index=False)
  
# right/left blink opening amplitude min (1)

main()

