import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import math
import statistics
import sys
#from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking3")
CH_DIR = os.path.join(DATA_DIR, "CH1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking4")

TIME_INTERVAL_DURATION = 60  #sec
#TIME_INTERVAL_DURATION = 180  #sec

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV"           ],
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
             ["D8r1_AP", "D8r2_AP", "D8r3_AP"],
             ["D8r4_AK", "D8r5_AK", "D8r6_AK"],
             ["D9r1_RE", "D9r2_RE", "D9r3_RE"],
             ["D9r4_SV", "D9r5_SV", "D9r6_SV"]
             ]

#filenames = [["D6r1_AE"]]

# Old features:
#           ['Saccade', 'Fixation',
#            'LeftPupilDiameter', 'RightPupilDiameter',
#            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
#            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
#            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
#            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
#            'HeadHeading', 'HeadPitch', 'HeadRoll']

new_features = ['SaccadesNumber', 'SaccadesTotalDuration',
                'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
                'SaccadesDurationMin', 'SaccadesDurationMax',
                'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
                'FixationDurationMin', 'FixationDurationMax',
                'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch', 'HeadRoll']


def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp >= ch_last_timestamp:
        return -1
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1


TI_df = pd.DataFrame()

for atco in filenames:
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(ET_DIR, 'ET_' + filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        negative_count = df['LeftBlinkOpeningAmplitude'].lt(0).sum()
        print(negative_count)
                       
        first_timestamp = df['UnixTimestamp'].tolist()[0]
                      
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_timestamps = scores_df['timestamp'].tolist()
        ch_first_timestamp = ch_timestamps[0]
        
        dif = first_timestamp - ch_first_timestamp
        if dif>0:
            ch_first_timestamp = first_timestamp
            
        number_of_ch_timestamps = len(ch_timestamps)
        ch_last_timestamp = ch_first_timestamp + 180*(number_of_ch_timestamps-1)
        
        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 
                       
        df = df[df['timeInterval']!=0]
        df = df[df['timeInterval']!=-1]
        
        timeIntervals = set(df['timeInterval'].tolist())
        number_of_time_intervals = len(timeIntervals)
                        
        SaccadesNumber = []
        SaccadesTotalDuration = []
        SaccadesDurationMean = []
        SaccadesDurationStd = []
        SaccadesDurationMedian = []
        SaccadesDurationMin = []
        SaccadesDurationMax = []

        #FixationNumber = []
        #FixationTotalDuration = []
        FixationDurationMean = []
        FixationDurationStd = []
        FixationDurationMedian = []
        FixationDurationMin = []
        FixationDurationMax = []
       
        #Add Saccade number, total duration and duration stats per period
        for ti in range(1, number_of_time_intervals+1):
            ti_df = df[df['timeInterval']==ti]
            
            if ti_df.empty:
                continue
            
            ti_saccades_df = ti_df[ti_df['Saccade']!=0]
            if ti_saccades_df.empty: # no saccade during the whole second, might be due to data loss
                # set the minimum
                saccades_total_duration = 1
                saccades_number = 1
                saccades_duration_mean = 1
                saccades_duration_std = 0
                saccades_duration_median = 1
                saccades_duration_min = 1
                saccades_duration_max = 1

            else:
                saccades_total_duration = len(ti_saccades_df.index)
                saccades_set = set(ti_saccades_df['Saccade'].tolist())
                saccades_number = len(saccades_set)
                saccades_duration = []
                for saccade in saccades_set:
                    saccade_df = ti_df[ti_df['Saccade']==saccade]
                    if not saccade_df.empty:
                        saccades_duration.append(len(saccade_df.index))
                
                saccades_duration_mean = statistics.mean(saccades_duration)
                saccades_duration_std = statistics.stdev(saccades_duration)
                saccades_duration_median = statistics.median(saccades_duration)
                saccades_duration_min = min(saccades_duration)
                saccades_duration_max = max(saccades_duration)

            ti_fixation_df = ti_df[ti_df['Fixation']!=0]

            if ti_fixation_df.empty: 
                print("No fixation during the whole second. Check the previous step.")
                sys.exit(1)

            else:
                #fixation_total_duration = len(ti_fixation_df.index)
                fixation_set = set(ti_fixation_df['Fixation'].tolist())
                #fixation_number = len(fixation_set)
                fixation_duration = []
                for fixation in fixation_set:
                    fixation_df = ti_df[ti_df['Fixation']==fixation]
                    if not fixation_df.empty:
                        fixation_duration.append(len(fixation_df.index))
            
                fixation_duration_mean = statistics.mean(fixation_duration)
                fixation_duration_std = statistics.stdev(fixation_duration) if len(fixation_duration)>1 else 0
                fixation_duration_median = statistics.median(fixation_duration)
                fixation_duration_min = min(fixation_duration)
                fixation_duration_max = max(fixation_duration)
            
            SaccadesNumber.extend([saccades_number]*TIME_INTERVAL_DURATION*250)
            SaccadesTotalDuration.extend([saccades_total_duration]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMean.extend([saccades_duration_mean]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationStd.extend([saccades_duration_std]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMedian.extend([saccades_duration_median]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMin.extend([saccades_duration_min]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMax.extend([saccades_duration_max]*TIME_INTERVAL_DURATION*250)
            
            FixationDurationMean.extend([fixation_duration_mean]*TIME_INTERVAL_DURATION*250)
            FixationDurationStd.extend([fixation_duration_std]*TIME_INTERVAL_DURATION*250)
            FixationDurationMedian.extend([fixation_duration_median]*TIME_INTERVAL_DURATION*250)
            FixationDurationMin.extend([fixation_duration_min]*TIME_INTERVAL_DURATION*250)
            FixationDurationMax.extend([fixation_duration_max]*TIME_INTERVAL_DURATION*250)
        
        df['SaccadesNumber'] = SaccadesNumber
        df['SaccadesTotalDuration'] = SaccadesTotalDuration
        df['SaccadesDurationMean'] = SaccadesDurationMean
        df['SaccadesDurationStd'] = SaccadesDurationStd
        df['SaccadesDurationMedian'] = SaccadesDurationMedian
        df['SaccadesDurationMin'] = SaccadesDurationMin
        df['SaccadesDurationMax'] = SaccadesDurationMax
        
        df['FixationDurationMean'] = FixationDurationMean
        df['FixationDurationStd'] = FixationDurationStd
        df['FixationDurationMedian'] = FixationDurationMedian
        df['FixationDurationMin'] = FixationDurationMin
        df['FixationDurationMax'] = FixationDurationMax
        
        df = df.drop('Saccade', axis=1)
        df = df.drop('Fixation', axis=1)
        
        row_num = len(df.index)
        df['ATCO'] = [filename[-2:]] * row_num
        df['Run'] = [run] * row_num
        run = run + 1    

        columns = ['ATCO'] + ['Run'] + ['timeInterval'] + ['UnixTimestamp'] + \
            ['SamplePerSecond'] + new_features
        df = df[columns]
        
        atco_df = pd.concat([atco_df, df], ignore_index=True)
    
    #####################################
    # Normalization per ATCO 
    # might cause data leakage
    '''
    scaler = preprocessing.MinMaxScaler()

    for feature in new_features:
        feature_lst = atco_df[feature].tolist()
        scaled_feature_lst = scaler.fit_transform(np.asarray(feature_lst).reshape(-1, 1))
        atco_df = atco_df.drop(feature, axis = 1)
        atco_df[feature] = scaled_feature_lst
    '''
    #####################################
    
    TI_df = pd.concat([TI_df, atco_df], ignore_index=True)

#print(TI_df.isnull().any().any())
#nan_count = TI_df.isna().sum()
#print(nan_count)

pd.set_option('display.max_columns', None)
#print(TI_df.head(1))

negative_count = TI_df['LeftBlinkOpeningAmplitude'].lt(0).sum()
print(negative_count)

full_filename = os.path.join(OUTPUT_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
TI_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
