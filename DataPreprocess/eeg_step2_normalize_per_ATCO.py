import os

import pandas as pd
from statistics import mean 

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EEG2")
OUTPUT_DIR = os.path.join(DATA_DIR, "EEG3")

metrics_list = ['workload', 'vigilance', 'stress']
#normalize for each participant (with all 3 runs as one set)

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV", "D2r3_KV"],
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

#filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"]]

def normilize(min, max, value):
    return ((value-min)/(max-min))

for atco in filenames:
    atco_df = pd.DataFrame()
    for filename in atco:
        print(filename)
        full_filename = os.path.join(INPUT_DIR, filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        atco_df = pd.concat([atco_df, df], ignore_index=True)
    
    atco_wl_min = atco_df['workload'].min()
    atco_wl_max = atco_df['workload'].max()
    
    atco_vig_min = atco_df['vigilance'].min()
    atco_vig_max = atco_df['vigilance'].max()
    
    atco_stress_min = atco_df['stress'].min()
    atco_stress_max = atco_df['stress'].max()

    
    for filename in atco:
        full_filename = os.path.join(INPUT_DIR, filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
    
        df['workload'] = df.apply(lambda row: normilize(atco_wl_min, atco_wl_max, row['workload']), axis=1)
        df['vigilance'] = df.apply(lambda row: normilize(atco_vig_min, atco_vig_max, row['vigilance']), axis=1)
        df['stress'] = df.apply(lambda row: normilize(atco_stress_min, atco_stress_max, row['stress']), axis=1)
    
        full_filename = os.path.join(OUTPUT_DIR, filename +  ".csv")
        df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
