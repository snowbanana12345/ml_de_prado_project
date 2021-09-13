import preprocessing.data_cleaning as dat_clean
import pandas as pd
import definitions
import os

# ------- file pathing and settings ------
root_dir = definitions.ROOTDIR
date = "20170131"
file_path = os.path.join(root_dir, "data_raw_tick", "ModelDepthProto_" + date + ".csv")

time_sampling_rule = "300S"
vol_bar_size = 20
tick_bar_size = 20
dollar_bar_size = 2000000

save_morning_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_morning_time_" + time_sampling_rule + "_sampled.csv")
save_afternoon_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_afternoon_time_" + time_sampling_rule + "_sampled.csv")
save_midnight_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_midnight_time_" + time_sampling_rule + "_sampled.csv")

save_morning_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_morning_volume_" + str(vol_bar_size) + "_sampled.csv")
save_afternoon_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")
save_midnight_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_midnight_volume_" + str(vol_bar_size) + "_sampled.csv")

save_morning_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_morning_tick_" + str(tick_bar_size) + "_sampled.csv")
save_afternoon_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_afternoon_tick_" + str(tick_bar_size) + "_sampled.csv")
save_midnight_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_midnight_tick_" + str(tick_bar_size) + "_sampled.csv")

save_morning_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_morning_dollar_" + str(dollar_bar_size) + "_sampled.csv")
save_afternoon_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_afternoon_dollar_" + str(dollar_bar_size) + "_sampled.csv")
save_midnight_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_midnight_dollar_" + str(dollar_bar_size) + "_sampled.csv")

# ----- clean up ------
unprocessed_tick_df = pd.read_csv(file_path)
morning_unprocessed_df, afternoon_unprocessed_df, midnight_unprocessed_df = dat_clean.morning_afternoon_midnight_split(unprocessed_tick_df)
morning_df = dat_clean.pre_process(morning_unprocessed_df)
afternoon_df = dat_clean.pre_process(afternoon_unprocessed_df)
midnight_df = dat_clean.pre_process(midnight_unprocessed_df)

# ----------- time sampling ---------
time_morning_df = dat_clean.time_sampling(morning_df, rule = time_sampling_rule, save = save_morning_path_time)
time_afternoon_df = dat_clean.time_sampling(afternoon_df, rule = time_sampling_rule, save = save_afternoon_path_time)
time_midnight_df = dat_clean.time_sampling(midnight_df, rule = time_sampling_rule, save = save_midnight_path_time)

# ----------- volume sampling -----------
volume_morning_df = dat_clean.volume_sampling(morning_df, bar_size = vol_bar_size, save = save_morning_path_vol)
volume_afternoon_df = dat_clean.volume_sampling(afternoon_df, bar_size = vol_bar_size, save = save_afternoon_path_vol)
volume_midnight_df = dat_clean.volume_sampling(midnight_df, bar_size = vol_bar_size, save = save_midnight_path_vol)

# ---------- tick sampling ---------
tick_morning_df = dat_clean.tick_sampling(morning_df, bar_size = tick_bar_size, save = save_morning_path_tick)
tick_afternoon_df = dat_clean.tick_sampling(afternoon_df, bar_size = tick_bar_size , save = save_afternoon_path_tick)
tick_midnight_df = dat_clean.tick_sampling(midnight_df, bar_size = tick_bar_size, save = save_midnight_path_tick)

# ---------- dollar sampling -------
dollar_morning_df = dat_clean.dollar_sampling(morning_df, bar_size = dollar_bar_size, save = save_morning_path_dollar)
dollar_afternoon_df = dat_clean.dollar_sampling(afternoon_df, bar_size = dollar_bar_size, save = save_afternoon_path_dollar)
dollar_midnight_df = dat_clean.dollar_sampling(midnight_df, bar_size = dollar_bar_size, save = save_midnight_path_dollar)
