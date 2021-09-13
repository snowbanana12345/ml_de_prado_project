from statsmodels.graphics.tsaplots import plot_acf
import definitions
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------- user inputs ----------------
root_dir = definitions.ROOTDIR
date = "20170126"

# ----- rule set 1 ------
#time_sampling_rule = "60S"
#vol_bar_size = 20
#tick_bar_size = 20
#dollar_bar_size = 400000

# ----- rule set 2 -----
time_sampling_rule = "300S"
vol_bar_size = 100
tick_bar_size = 100
dollar_bar_size = 2000000

plot_morning_time = date + "_morning_" + time_sampling_rule + "_time"
plot_afternoon_time = date + "_afternoon_" + time_sampling_rule + "_time"
plot_morning_vol = date + "_morning_" + str(vol_bar_size) + "_volume"
plot_afternoon_vol = date + "_afternoon_" + str(vol_bar_size) + "_volume"
plot_morning_tick = date + "_morning_" + str(tick_bar_size) + "_tick"
plot_afternoon_tick = date + "_afternoon_" + str(tick_bar_size) + "_tick"
plot_morning_dollar = date + "_morning_" + str(dollar_bar_size) + "_dollar"
plot_afternoon_dollar = date + "_afternoon_" + str(dollar_bar_size) + "_dollar"

save_morning_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_morning_time_" + time_sampling_rule + "_sampled.csv")
save_afternoon_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_afternoon_time_" + time_sampling_rule + "_sampled.csv")
save_midnight_path_time = os.path.join(root_dir, "data_bar", "time_sampled", date + "_midnight_time_" + time_sampling_rule + "_sampled.csv")

save_morning_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_morning_volume_" + str(vol_bar_size) + "_sampled.csv")
save_afternoon_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")
save_midnight_path_vol = os.path.join(root_dir, "data_bar", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")

save_morning_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_morning_tick_" + str(tick_bar_size) + "_sampled.csv")
save_afternoon_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_afternoon_tick_" + str(tick_bar_size) + "_sampled.csv")
save_midnight_path_tick = os.path.join(root_dir, "data_bar", "tick_sampled", date + "_midnight_tick_" + str(tick_bar_size) + "_sampled.csv")

save_morning_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_morning_dollar_" + str(dollar_bar_size) + "_sampled.csv")
save_afternoon_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_afternoon_dollar_" + str(dollar_bar_size) + "_sampled.csv")
save_midnight_path_dollar = os.path.join(root_dir, "data_bar", "dollar_sampled", date + "_midnight_dollar_" + str(dollar_bar_size) + "_sampled.csv")

# ---------read data -----------
time_morning_df = pd.read_csv(save_morning_path_time)
time_afternoon_df = pd.read_csv(save_afternoon_path_time)
time_midnight_df = pd.read_csv(save_midnight_path_time)

volume_morning_df = pd.read_csv(save_morning_path_vol)
volume_afternoon_df = pd.read_csv(save_afternoon_path_vol)
volume_midnight_df = pd.read_csv(save_midnight_path_vol)

tick_morning_df = pd.read_csv(save_morning_path_tick)
tick_afternoon_df = pd.read_csv(save_afternoon_path_tick)
tick_midnight_df = pd.read_csv(save_midnight_path_tick)

dollar_morning_df = pd.read_csv(save_morning_path_dollar)
dollar_afternoon_df = pd.read_csv(save_afternoon_path_dollar)
dollar_midnight_df = pd.read_csv(save_midnight_path_dollar)

# ------- auto correlations --------
time_morning_vvap = time_morning_df["VVAP"].replace(0, np.nan).interpolate()
time_afternoon_vvap = time_afternoon_df["VVAP"].replace(0, np.nan).interpolate()
plot_acf(time_morning_vvap.diff().dropna(), title = plot_morning_time + "_diff_auto_correlation ", lags = 10)
plot_acf(time_afternoon_vvap.diff().dropna(), title = plot_afternoon_time + "_diff_auto_correlation ", lags = 10)
plot_acf(volume_morning_df["VVAP"].diff().dropna(), title = plot_morning_vol + "_diff_auto_correlation ", lags = 10)
plot_acf(volume_afternoon_df["VVAP"].diff().dropna(), title = plot_afternoon_vol + "_diff_auto_correlation ", lags = 10)
plot_acf(tick_morning_df["VVAP"].diff().dropna(), title = plot_morning_tick + "_diff_auto_correlation ", lags = 10)
plot_acf(tick_afternoon_df["VVAP"].diff().dropna(), title = plot_afternoon_tick + "_diff_auto_correlation ", lags = 10)
morning_dollar_vvap = dollar_bar_size / dollar_morning_df["volume"]
afternoon_dollar_vvap = dollar_bar_size / dollar_afternoon_df["volume"]
plot_acf(morning_dollar_vvap.diff().dropna(), title = plot_morning_dollar + "_diff_auto_correlation ", lags = 10)
plot_acf(afternoon_dollar_vvap.diff().dropna(), title = plot_afternoon_dollar + "_diff_auto_correlation ", lags = 10)
plt.show()



