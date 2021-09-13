import pandas as pd
import definitions
import os
import matplotlib.pyplot as plt
import plotter.plotter as pltr


# ----------- user inputs ----------------
root_dir = definitions.ROOTDIR
date = "20170201"

time_sampling_rule = "300S"
vol_bar_size = 100
tick_bar_size = 100
dollar_bar_size = 2000000

plot_morning_vol = date + "_morning_" + str(vol_bar_size) + "_volume"
plot_afternoon_vol = date + "_afternoon_" + str(vol_bar_size) + "_volume"
plot_midnight_vol = date + "_midnight_" + str(vol_bar_size) + "_volume"

plot_morning_tick = date + "_morning_" + str(vol_bar_size) + "_tick"
plot_afternoon_tick = date + "_afternoon_" + str(vol_bar_size) + "_tick"
plot_midnight_tick = date + "_midnight_" + str(vol_bar_size) + "_tick"

save_morning_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_morning_volume_" + str(vol_bar_size) + "_sampled.csv")
save_afternoon_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_afternoon_volume_" + str(vol_bar_size) + "_sampled.csv")
save_midnight_path_vol = os.path.join(root_dir, "data_bar_wlimit", "volume_sampled", date + "_midnight_volume_" + str(vol_bar_size) + "_sampled.csv")

save_morning_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_morning_tick_" + str(tick_bar_size) + "_sampled.csv")
save_afternoon_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_afternoon_tick_" + str(tick_bar_size) + "_sampled.csv")
save_midnight_path_tick = os.path.join(root_dir, "data_bar_wlimit", "tick_sampled", date + "_midnight_tick_" + str(tick_bar_size) + "_sampled.csv")

image_folders_path = os.path.join(root_dir, "data_plots", "raw_bar_wlimit_coarse")

win_x_inches = 16
win_y_inches = 8

image_folder_path = os.path.join(image_folders_path, date)
if not os.path.isdir(image_folder_path):
    os.mkdir(image_folder_path)

volume_morning_plot_path = os.path.join(image_folder_path, "volume_morning.png")
volume_afternoon_plot_path = os.path.join(image_folder_path, "volume_afternoon.png")
volume_midnight_plot_path = os.path.join(image_folder_path, "volume_midnight.png")

tick_morning_plot_path = os.path.join(image_folder_path, "tick_morning.png")
tick_afternoon_plot_path = os.path.join(image_folder_path, "tick_afternoon.png")
tick_midnight_plot_path = os.path.join(image_folder_path, "tick_midnight.png")

# ---------read data -----------
#volume_morning_df = pd.read_csv(save_morning_path_vol)
#volume_afternoon_df = pd.read_csv(save_afternoon_path_vol)
#volume_midnight_df = pd.read_csv(save_midnight_path_vol)

tick_morning_df = pd.read_csv(save_morning_path_tick)
tick_afternoon_df = pd.read_csv(save_afternoon_path_tick)
tick_midnight_df = pd.read_csv(save_midnight_path_tick)

# --------- plotting volume sampled -----------
"""
plt.figure()
pltr.plot_xy_multi(volume_morning_df["volume_time"], [volume_morning_df["VVAP"], volume_morning_df["average_bid"], volume_morning_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "volume traded", y_label = "VVAP", title = plot_morning_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_multi(volume_afternoon_df["volume_time"], [volume_afternoon_df["VVAP"], volume_afternoon_df["average_bid"], volume_afternoon_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "volume traded", y_label = "VVAP", title = plot_afternoon_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_multi(volume_midnight_df["volume_time"], [volume_midnight_df["VVAP"], volume_midnight_df["average_bid"], volume_midnight_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "volume traded", y_label = "VVAP", title = plot_midnight_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_midnight_plot_path, win_x_inches, win_y_inches)
"""
# --------- plotting tick sampled -----------

plt.figure()
pltr.plot_xy_multi(tick_morning_df["tick_time"], [tick_morning_df["VVAP"], tick_morning_df["average_bid"], tick_morning_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "ticks", y_label = "price", title = plot_morning_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_multi(tick_afternoon_df["tick_time"], [tick_afternoon_df["VVAP"], tick_afternoon_df["average_bid"], tick_afternoon_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "ticks", y_label = "price", title = plot_afternoon_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_multi(tick_midnight_df["tick_time"], [tick_midnight_df["VVAP"], tick_midnight_df["average_bid"], tick_midnight_df["average_ask"]],
                   labels = ["VVAP", "average bid", "average ask"],x_label = "ticks", y_label = "price", title = plot_midnight_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_midnight_plot_path, win_x_inches, win_y_inches)


