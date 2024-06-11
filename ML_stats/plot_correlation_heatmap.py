import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 180

def main():
    
    if TIME_INTERVAL_DURATION == 60:
        filename = "ML_features_1min.csv"
    else:
        filename = "ML_features_3min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df.drop('ATCO', axis=1)
    
    print(data_df.info())
    print(data_df.head())
    
    features = data_df.columns
    print(len(features))
  
    
    plt.figure(figsize=(20, 18))
    
    # plotting correlation heatmap
    corr_matrix = data_df.corr()
    print(corr_matrix)
    
    print(len(corr_matrix))
    sns.heatmap(corr_matrix, annot=False)
  
    #fig.savefig('correlations.png', dpi=800)
    # displaying heatmap 
    plt.show()
    
main()