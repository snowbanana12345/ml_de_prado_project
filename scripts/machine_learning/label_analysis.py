import definitions
import os
import pandas as pd
import matplotlib.pyplot as plt
import machine_learning.feature_engineering as fe


root_dir = definitions.ROOTDIR
bar_data_folder_path = os.path.join(root_dir, "data_bar")
bar_file_paths = [os.path.join(bar_data_folder_path, "volume_sampled", "20170125_morning_volume_20_sampled.csv"),
                        os.path.join(bar_data_folder_path, "volume_sampled", "20170126_morning_volume_20_sampled.csv"),
                        os.path.join(bar_data_folder_path, "volume_sampled", "20170127_morning_volume_20_sampled.csv"),
                        os.path.join(bar_data_folder_path, "volume_sampled", "20170131_morning_volume_20_sampled.csv")]

bar_df_lst = [pd.read_csv(f_path) for f_path in bar_file_paths]


look_ahead = 30
threshold = 10

bar_df_lst = [fe.create_delta_labels(train_bar_df, look_ahead, threshold) for train_bar_df in bar_df_lst]
bar_df = pd.concat(bar_df_lst, ignore_index = True, axis = 0)
print(bar_df["delta_label"].value_counts())

