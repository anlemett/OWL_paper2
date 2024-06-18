import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import math

#import sys

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking4")
EEG_DIR = os.path.join(DATA_DIR, "EEG4")
ML_DIR = os.path.join(DATA_DIR, "MLInput")


TIME_INTERVAL_DURATION = 60
WINDOW_SIZE = 250 * TIME_INTERVAL_DURATION

features = ['FixationNumber', 'FixationTotalDuration',
            'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
            'FixationDurationMin', 'FixationDurationMax',
            'SaccadesNumber', 'SaccadesTotalDuration',
            'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
            'SaccadesDurationMin', 'SaccadesDurationMax',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll'
            ]

ATCOs = ['MO', 'EI', 'KV', 'UO', 'KB', 'PF', 'AL', 'IH', 'RI',
         'JO', 'AE', 'HC', 'LS', 'ML', 'AP', 'AK', 'RE', 'SV']

def get_TS_np(features):
    
    window_size = 250 * TIME_INTERVAL_DURATION
    number_of_features = len(features)
    number_of_features = number_of_features + 1  # + ATCO
    
    # TS_np shape (a,b,c):
    # a - number of time intervals, b - number of measures per time interval (WINDOW_SIZE),
    # c - number of features
    
    # we squeeze to 0 the dimension which we do not know and
    # to which we want to append
    TS_np = np.zeros(shape=(0, window_size, number_of_features))
    
    all_WL_scores = []
    all_Vig_scores = []
    all_Stress_scores = []
    
    #**************************************
    print("Reading Eye Tracking data")
    full_filename = os.path.join(ET_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
    et_df = pd.read_csv(full_filename, sep=' ')
    
    #print(et_df.isnull().any().any())
    #The output shows the number of NaN values in each column of the data frame
    #nan_count = et_df.isna().sum()
    #print(nan_count)

    print("Reading EEG data")
    full_filename = os.path.join(EEG_DIR, "EEG_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
    eeg_df = pd.read_csv(full_filename, sep=' ')
  
    dim1_idx = 0

    atco_num = 0
    for atco in ATCOs:
        print(atco)
        atco_num = atco_num + 1
        et_atco_df = et_df[et_df['ATCO']==atco]
        eeg_atco_df = eeg_df[eeg_df['ATCO']==atco]
        
        if et_atco_df.empty or eeg_atco_df.empty:
            continue
        
        for run in range(1,4):
            et_run_df = et_atco_df[et_atco_df['Run']==run]
            eeg_run_df = eeg_atco_df[eeg_atco_df['Run']==run]
            
            if et_run_df.empty or eeg_run_df.empty:
                continue
        
            number_of_time_intervals = len(eeg_run_df['timeInterval'].tolist())
        
            run_TS_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
            run_WL_scores = []
            run_Vig_scores = []
            run_Stress_scores = []
            
            print(number_of_time_intervals)
            dim1_idx = 0
            for ti in range(1, number_of_time_intervals+1):
                et_ti_df = et_run_df[et_run_df['timeInterval']==ti]
                eeg_ti_df = eeg_run_df[eeg_run_df['timeInterval']==ti]
                
                ti_WL_score_lst = eeg_ti_df['WorkloadMean'].tolist()
                ti_Vig_score_lst = eeg_ti_df['VigilanceMean'].tolist()
                ti_Stress_score_lst = eeg_ti_df['StressMean'].tolist()
                if math.isnan(ti_WL_score_lst[0]):
                    continue
                if math.isnan(ti_Vig_score_lst[0]):
                    continue                
                if math.isnan(ti_Stress_score_lst[0]):
                    continue
                if et_ti_df.empty:
                    continue
                
                ti_WL_score = ti_WL_score_lst[0]
                ti_Vig_score = ti_Vig_score_lst[0]
                ti_Stress_score = ti_Stress_score_lst[0]
                
                dim2_idx = 0
                for index, row in et_ti_df.iterrows():
                    #exclude ATCO, Run, timeInterval, UnixTimestamp, SamplePerSecond
                    lst_of_features = row.values.tolist()[5:]
                    run_TS_np[dim1_idx, dim2_idx] = [atco_num] + lst_of_features
                    dim2_idx = dim2_idx + 1
                    
                run_WL_scores.append(ti_WL_score)
                run_Vig_scores.append(ti_Vig_score)
                run_Stress_scores.append(ti_Stress_score)
                        
                dim1_idx = dim1_idx + 1
                
            if dim1_idx < number_of_time_intervals:
                run_TS_np = run_TS_np[:dim1_idx]
                
            TS_np = np.append(TS_np, run_TS_np, axis=0)
            all_WL_scores.extend(run_WL_scores)
            all_Vig_scores.extend(run_Vig_scores)
            all_Stress_scores.extend(run_Stress_scores)

    all_scores = np.array((all_WL_scores, all_Vig_scores, all_Stress_scores))
    return (TS_np, all_scores)

(TS_np, scores) = get_TS_np(features)

#print(np.isnan(TS_np).any())

print(TS_np.shape) # 60 -> (1731, 15000, 27)

print(len(scores))

# Reshape the 3D array to 2D
TS_np_reshaped = TS_np.reshape(TS_np.shape[0], -1)

# Save the 2D array to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
np.savetxt(full_filename, TS_np_reshaped, delimiter=" ")

# Save scores to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")
np.savetxt(full_filename, np.asarray(scores) , delimiter=" ")


