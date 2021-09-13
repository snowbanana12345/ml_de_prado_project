import preprocessing.data_cleaning as dat_clean
import pandas as pd
import definitions
import os
import numpy as np


# ------- file pathing and settings ------
root_dir = definitions.ROOTDIR
date = "20170217"
file_path = os.path.join(root_dir, "data_raw_tick", "ModelDepthProto_" + date + ".csv")

time_sampling_rule = "300S"
vol_bar_size = 20
tick_bar_size = 20
dollar_bar_size = 2000000

save_morning_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_morning_volume_" + str(vol_bar_size) + "_sampled.csv")
save_afternoon_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")
save_midnight_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_midnight_volume_" + str(vol_bar_size) + "_sampled.csv")

save_morning_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_morning_tick_" + str(tick_bar_size) + "_sampled.csv")
save_afternoon_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_afternoon_tick_" + str(tick_bar_size) + "_sampled.csv")
save_midnight_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_midnight_tick_" + str(tick_bar_size) + "_sampled.csv")

# ----- clean up ------
unprocessed_tick_df = pd.read_csv(file_path)
unprocessed_tick_df["best_ask"] = np.minimum.reduce([unprocessed_tick_df["ask1p"], unprocessed_tick_df["ask2p"],unprocessed_tick_df["ask3p"]
                                                        , unprocessed_tick_df["ask4p"], unprocessed_tick_df["ask5p"]])
unprocessed_tick_df["best_bid"] = np.maximum.reduce([unprocessed_tick_df["bid1p"], unprocessed_tick_df["bid2p"],unprocessed_tick_df["bid3p"]
                                                    , unprocessed_tick_df["bid4p"], unprocessed_tick_df["bid5p"]])

morning_unprocessed_df, afternoon_unprocessed_df, midnight_unprocessed_df = dat_clean.morning_afternoon_midnight_split(unprocessed_tick_df)
morning_df = dat_clean.pre_process(morning_unprocessed_df)
afternoon_df = dat_clean.pre_process(afternoon_unprocessed_df)
midnight_df = dat_clean.pre_process(midnight_unprocessed_df)

# ----------- volume sampling -----------
volume_morning_df = dat_clean.volume_sampling_limit_book(morning_df, bar_size = vol_bar_size, save = save_morning_path_vol)
volume_afternoon_df = dat_clean.volume_sampling_limit_book(afternoon_df, bar_size = vol_bar_size, save = save_afternoon_path_vol)
volume_midnight_df = dat_clean.volume_sampling_limit_book(midnight_df, bar_size = vol_bar_size, save = save_midnight_path_vol)

# ---------- tick sampling ---------
tick_morning_df = dat_clean.tick_sampling_limit_book(morning_df, bar_size = tick_bar_size, save = save_morning_path_tick)
tick_afternoon_df = dat_clean.tick_sampling_limit_book(afternoon_df, bar_size = tick_bar_size , save = save_afternoon_path_tick)
tick_midnight_df = dat_clean.tick_sampling_limit_book(midnight_df, bar_size = tick_bar_size, save = save_midnight_path_tick)
