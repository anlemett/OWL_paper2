import warnings
warnings.filterwarnings('ignore')

import os
import rdata

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "CorrAnalysis")

full_filename = os.path.join(DATA_DIR, "Correlation_Anastasia.RData")

dict1 = rdata.read_rda(full_filename)

#print(dict1.keys())

features_to_drop = [
 'Saccades.Duration.Min.wg',
 'Fixation.Duration.Min.wg',
 'Left.Blink.Closing.Amplitude.Min.wg',
 'Left.Blink.Opening.Amplitude.Min.wg',
 'Left.Blink.Closing.Speed.Min.wg',
 'Left.Blink.Opening.Speed.Min.wg',
 'Right.Blink.Closing.Amplitude.Min.wg',
 'Right.Blink.Opening.Amplitude.Min.wg',
 'Right.Blink.Closing.Speed.Min.wg', 
 'Right.Blink.Opening.Speed.Min.wg',
 'Left.Blink.Closing.Amplitude.Median.wg',
 'Left.Blink.Opening.Amplitude.Median.wg',
 'Left.Blink.Closing.Speed.Median.wg',
 'Left.Blink.Opening.Speed.Median.wg',
 'Right.Blink.Closing.Amplitude.Median.wg',
 'Right.Blink.Opening.Amplitude.Median.wg',
 'Right.Blink.Closing.Speed.Median.wg',
 'Right.Blink.Opening.Speed.Median.wg'
]

dict2 = dict1["res1_ATCO"]
#print(dict2.keys())

corr_matrix_arr = dict2["rwg"]
#print(type(corr_matrix_arr))
corr_matrix_arr = corr_matrix_arr.drop_sel(dim_0=features_to_drop, dim_1=features_to_drop)
#print(corr_matrix_arr.dim_0)

corr_matrix_df = corr_matrix_arr.to_dataframe(name="corr_matrix")

full_filename = os.path.join(DATA_DIR, "corr_matrix_1min.csv")
corr_matrix_df.to_csv(full_filename)

p_values_arr = dict2["pwg"]
p_values_arr = p_values_arr.drop_sel(dim_0=features_to_drop, dim_1=features_to_drop)
p_values_df = p_values_arr.to_dataframe(name="p_values")

full_filename = os.path.join(DATA_DIR, "p_values_1min.csv")
p_values_df.to_csv(full_filename)


dict2 = dict1["res3_ATCO"]
#print(dict2.keys())

corr_matrix_arr = dict2["rwg"]
corr_matrix_arr = corr_matrix_arr.drop_sel(dim_0=features_to_drop, dim_1=features_to_drop)
corr_matrix_df = corr_matrix_arr.to_dataframe(name="corr_matrix")

full_filename = os.path.join(DATA_DIR, "corr_matrix_3min.csv")
corr_matrix_df.to_csv(full_filename)

p_values_arr = dict2["pwg"]
p_values_arr = p_values_arr.drop_sel(dim_0=features_to_drop, dim_1=features_to_drop)
p_values_df = p_values_arr.to_dataframe(name="p_values")

full_filename = os.path.join(DATA_DIR, "p_values_3min.csv")
p_values_df.to_csv(full_filename)
