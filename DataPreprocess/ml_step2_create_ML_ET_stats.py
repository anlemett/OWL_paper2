import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 180

saccade_fixation = [
            'Saccades Number', 'Saccades Total Duration',
            'Saccades Duration Mean', 'Saccades Duration Std', 'Saccades Duration Median',
            'Saccades Duration Min', 'Saccades Duration Max',
            'Fixation Number', 'Fixation Total Duration',
            'Fixation Duration Mean', 'Fixation Duration Std', 'Fixation Duration Median',
            'Fixation Duration Min', 'Fixation Duration Max'
            ]

old_features = [
            'Left Pupil Diameter', 'Right Pupil Diameter',
            'Left Blink Closing Amplitude', 'Left Blink Opening Amplitude',
            'Left Blink Closing Speed', 'Left Blink Opening Speed',
            'Right Blink Closing Amplitude', 'Right Blink Opening Amplitude',
            'Right Blink Closing Speed', 'Right Blink Opening Speed',
            'Head Heading', 'Head Pitch', 'Head Roll']

statistics = ['Mean', 'Std', 'Min', 'Max', 'Median']

features = []
for feature in saccade_fixation:
    features.append(feature)
for stat in statistics:
    for feature in old_features:
        new_feature = feature + ' ' + stat
        features.append(new_feature)

np.random.seed(0)


def featurize_data(x_data):
    """
    :param x_data: numpy array of shape
    (number_of_timeintervals, number_of_timestamps, number_of_features)
    where number_of_timestamps == TIME_INTERVAL_DURATION*250

    :return: featurized numpy array of shape
    (number_of_timeintervals, number_of_new_features)
    """
    print("Input shape before feature union:", x_data.shape)
    
    new_data = x_data[:,0,:15]

    feature_to_featurize = x_data[:,:,15:]
    
    mean = np.mean(feature_to_featurize, axis=-2)
    std = np.std(feature_to_featurize, axis=-2)
    median = np.median(feature_to_featurize, axis=-2)
    min = np.min(feature_to_featurize, axis=-2)
    max = np.max(feature_to_featurize, axis=-2)

    featurized_data = np.concatenate([
        mean,    
        std,     
        min,
        max, 
        median
    ], axis=-1)

    new_data = np.concatenate((new_data, featurized_data), axis=1)
    
    print("Shape after feature union, before classification:", new_data.shape)
    return new_data


def main():
    
    if TIME_INTERVAL_DURATION == 60:
        full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
    else:
        full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
        
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")
    
    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # 60 -> (1731, 15000, 28), 180 -> (667, 45000, 28)
    if TIME_INTERVAL_DURATION == 60:
        TS_np = TS_np.reshape((1731, 15000, 28))
    else:
        TS_np = TS_np.reshape((667, 45000, 28))
    
    X_featurized = featurize_data(TS_np)
    
    data_df = pd.DataFrame.from_records(X_featurized, columns=['ATCO'] + features)
    
    if TIME_INTERVAL_DURATION == 60:
        filename = "ML_features_1min.csv"
    else:
        filename = "ML_features_3min.csv"
    
    
    # Remove features with zero std

    features_to_remove = ["Saccades Duration Min", "Fixation Duration Min",
                          "Left Blink Closing Amplitude Min",
                          "Left Blink Opening Amplitude Min",
                          "Left Blink Closing Speed Min",
                          "Left Blink Opening Speed Min",
                          "Right Blink Closing Amplitude Min",
                          "Right Blink Opening Amplitude Min",
                          "Right Blink Closing Speed Min",
                          "Right Blink Opening Speed Min",
                          "Left Blink Closing Amplitude Median",
                          "Left Blink Opening Amplitude Median",
                          "Left Blink Closing Speed Median",
                          "Left Blink Opening Speed Median",
                          "Right Blink Closing Amplitude Median",
                          "Right Blink Opening Amplitude Median",
                          "Right Blink Closing Speed Median",
                          "Right Blink Opening Speed Median"
                      ]

    data_df = data_df.drop(features_to_remove, axis=1)
    
    full_filename = os.path.join(ML_DIR, filename)
    data_df.to_csv(full_filename, sep =" ", header=True, index=False)

main()
