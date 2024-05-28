import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import math
from statistics import mean, median
#import sys

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
#EEG_DIR = os.path.join(DATA_DIR, "EEG3") #normalized data
EEG_DIR = os.path.join(DATA_DIR, "EEG2") #raw data
CH_DIR = os.path.join(DATA_DIR, "CH1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EEG4")

TIME_INTERVAL_DURATION = 60  #sec

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV", "D2r2_KV"],
             ["D2r4_UO", "D2r5_UO", "D2r6_UO"],
             ["D3r1_KB", "D3r2_KB", "D3r3_KB"],
             ["D3r4_PF", "D3r5_PF", "D3r6_PF"],
             ["D4r1_AL", "D4r2_AL", "D4r3_AL"],
             ["D4r4_IH", "D4r5_IH", "D4r6_IH"],
             ["D5r1_RI", "D5r2_RI", "D5r3_RI"],
             ["D5r4_JO", "D5r5_JO", "D5r6_JO"],
             ["D6r1_AE", "D6r2_AE", "D6r3_AE"],
             ["D6r4_HC", "D6r5_HC", "D6r6_HC"],
             ["D7r1_LS", "D7r2_LS", "D7r3_LS"],
             ["D7r4_ML", "D7r5_ML", "D7r6_ML"],
             [           "D8r5_AK", "D8r6_AK"],
             ["D9r1_RE", "D9r2_RE", "D9r3_RE"],
             ["D9r4_SV", "D9r5_SV", "D9r6_SV"]
             ]


def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp > ch_last_timestamp:
        return 0
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1


ML_df = pd.DataFrame()

for atco in filenames:
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(EEG_DIR, filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_first_timestamp = scores_df['timestamp'].loc[0]
        ch_last_timestamp = scores_df['timestamp'].tolist()[-1]

        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 

        df = df[df['timeInterval']!=0]
        
        eeg_timeintervals = set(df['timeInterval'].tolist())
        number_of_time_intervals = len(eeg_timeintervals)
        
        for ti in range (1, number_of_time_intervals + 1):
            ti_df = df[df['timeInterval']==ti]
            if ti_df.empty:
                 ti_wl_mean = None
                 ti_wl_median = None
                 ti_vig_mean = None
                 ti_vig_median = None
                 ti_stress_mean = None
                 ti_stress_median = None
            else:
                ti_wl_mean = mean(ti_df.dropna()['workload'].tolist())
                ti_wl_median = median(ti_df.dropna()['workload'].tolist())
                ti_vig_mean = mean(ti_df.dropna()['vigilance'].tolist())
                ti_vig_median = median(ti_df.dropna()['vigilance'].tolist())
                ti_stress_mean = mean(ti_df.dropna()['stress'].tolist())
                ti_stress_median = median(ti_df.dropna()['stress'].tolist())

                
            new_row = {'ATCO': filename[-2:], 'Run': run, 'timeInterval': ti,
                       'WorkloadMean': ti_wl_mean, 'WorkloadMedian': ti_wl_median,
                       'VigilanceMean': ti_vig_mean, 'VigilanceMedian': ti_vig_median,
                       'StressMean': ti_stress_mean, 'StressMedian': ti_stress_median,
                       }

            ML_df = pd.concat([ML_df, pd.DataFrame([new_row])], ignore_index=True)
                
        run = run + 1
        
full_filename = os.path.join(OUTPUT_DIR, "EEG_all_" + str (TIME_INTERVAL_DURATION) + ".csv")
ML_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
