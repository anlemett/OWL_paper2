import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
from statistics import mean 

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EEG1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EEG2")


def getUnixTimestampS(ts_ms):
    return int(ts_ms/1000)

def getValuesPerSecond(timestamps_dict, unix_timestamp):
    return timestamps_dict[unix_timestamp]

filenames = ["D1r1_MO", "D1r2_MO", "D1r3_MO",
             "D1r4_EI", "D1r5_EI", "D1r6_EI",
             "D2r1_KV", "D2r2_KV", "D2r3_KV",
             "D2r4_UO", "D2r5_UO", "D2r6_UO",
             "D3r1_KB", "D3r2_KB", "D3r3_KB",
             "D3r4_PF", "D3r5_PF", "D3r6_PF",
             "D4r1_AL", "D4r2_AL", "D4r3_AL",
             "D4r4_IH", "D4r5_IH", "D4r6_IH",
             "D5r1_RI", "D5r2_RI", "D5r3_RI",
             "D5r4_JO", "D5r5_JO", "D5r6_JO",
             "D6r1_AE", "D6r2_AE", "D6r3_AE",
             "D6r4_HC", "D6r5_HC", "D6r6_HC",
             "D7r1_LS", "D7r2_LS", "D7r3_LS",
             "D7r4_ML", "D7r5_ML", "D7r6_ML",
                        "D8r5_AK", "D8r6_AK",
             "D9r1_RE", "D9r2_RE", "D9r3_RE",
             "D9r4_SV", "D9r5_SV", "D9r6_SV"
             ]

#filenames = ["D1r1_MO", "D1r2_MO", "D1r3_MO"]

for filename in filenames:
    print(filename)
    full_filename = os.path.join(INPUT_DIR, filename +  ".csv")
    df = pd.read_csv(full_filename, sep=';')
    df.sort_values(['calculatedAt'], ascending=[True], inplace=True)
    df.reset_index(inplace=True)

    df['UnixTimestamp'] = df.apply(lambda row: getUnixTimestampS(row['calculatedAt']), axis=1)
    
    first_timestamp = df['UnixTimestamp'].loc[0]
    last_timestamp = df['UnixTimestamp'].loc[len(df.index)-1]
    
    print(first_timestamp)
    print(last_timestamp)
    
    new_df = pd.DataFrame(columns=['UnixTimestamp', 'workload', 'vigilance', 'stress'])

    for ts in range(first_timestamp, last_timestamp + 1):
 
        ts_df = df[df['UnixTimestamp']==ts]
 
        if ts_df.empty:
            print(filename + ": empty second")
            continue
        
        eeg_wl_av = mean(ts_df['workload'].tolist())
        
        vig_lst = ts_df['vigilance'].tolist()
        
        for i in range(1, len(vig_lst)): #0 element seems to be fine in all files
            if vig_lst[i]>100:
                vig_lst[i] = vig_lst[i-1] 
        
        eeg_vig_av = mean(vig_lst)
        
        eeg_stress_av = mean(ts_df['stress'].tolist())
        
        
        # append the row
        new_row = {'UnixTimestamp': ts, 'workload': eeg_wl_av,
                   'vigilance': eeg_vig_av, 'stress': eeg_stress_av}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
 
    full_filename = os.path.join(OUTPUT_DIR, filename +  ".csv")
    new_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
