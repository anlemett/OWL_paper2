import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import itertools

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EyeTracking1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking2")

metrics_list = ['Saccade', 'Fixation',
                'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch',	'HeadRoll']
 
def getValuesPerSecond(timestamps_dict, unix_timestamp):
    return timestamps_dict[unix_timestamp]

filenames = ["D1r1_MO", "D1r2_MO", "D1r3_MO",
             "D1r4_EI", "D1r5_EI", "D1r6_EI",
             "D2r1_KV", "D2r2_KV",
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
             "D8r1_AP", "D8r2_AP", "D8r3_AP",
             "D8r4_AK", "D8r5_AK", "D8r6_AK",
             "D9r1_RE", "D9r2_RE", "D9r3_RE",
             "D9r4_SV", "D9r5_SV", "D9r6_SV"
             ]
#for testing
filenames = ["D1r1_MO"]

for filename in filenames:
    full_filename = os.path.join(INPUT_DIR, "ET_" + filename +  ".csv")
    df = pd.read_csv(full_filename, sep=' ')

    sample_per_second_lst = []
    first_timestamp = df['UnixTimestamp'].loc[0]
    last_timestamp = df['UnixTimestamp'].loc[len(df.index)-1]
 
    for ts in range(first_timestamp, last_timestamp + 1):
     
        ts_df = df[df['UnixTimestamp']==ts]
     
        if ts_df.empty:
            print(filename + ": empty second")
     
        for i in range(0, len(ts_df.index)):
            sample_per_second_lst.extend([i+1])
            
            
    df['SamplePerSecond'] = sample_per_second_lst

    columns = ['UnixTimestamp'] + ['SamplePerSecond'] + metrics_list
    df = df[columns]
    
    print(len(df.index))

    full_filename = os.path.join(OUTPUT_DIR, "ET_" + filename +  ".csv")
    df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
  