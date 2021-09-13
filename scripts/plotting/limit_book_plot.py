import pandas as pd
import definitions
import os
import matplotlib.pyplot as plt
import plotter.plotter as pltr


# ----------- user inputs ----------------
root_dir = definitions.ROOTDIR
date = "20170125"

time_sampling_rule = "300S"
vol_bar_size = 100
tick_bar_size = 100
dollar_bar_size = 2000000

save_morning_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_morning_volume_" + str(vol_bar_size) + "_sampled.csv")
save_afternoon_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")
save_midnight_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_midnight_volume_" + str(vol_bar_size) + "_sampled.csv")


# ---------read data -----------
volume_morning_df = pd.read_csv(save_morning_path_vol)
volume_afternoon_df = pd.read_csv(save_afternoon_path_vol)
volume_midnight_df = pd.read_csv(save_midnight_path_vol)

# --------- plotting volume sampled -----------
plt.figure()
pltr.plot_xy_multi(volume_morning_df["volume_time"], [volume_morning_df["open_ask"], volume_morning_df["low_ask"]
    , volume_morning_df["high_ask"], volume_morning_df["close_ask"]],
    labels = ["open_ask", "low_ask", "high_ask", "close_ask"], x_label = "volume traded", y_label = "VVAP", title = "morning")
plt.grid(linewidth = 1)

plt.figure()
pltr.plot_xy(volume_morning_df["volume_time"], volume_morning_df["average_bid_ask_spread"], x_label = "volume traded"
             ,y_label = "average_bid_ask_spread", title = "morning")

plt.show()