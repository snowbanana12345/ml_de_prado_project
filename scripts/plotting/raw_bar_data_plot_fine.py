import pandas as pd
import definitions
import os
import matplotlib.pyplot as plt
import plotter.plotter as pltr


# ----------- user inputs ----------------
root_dir = definitions.ROOTDIR
date = "20170228"

time_sampling_rule = "60S"
vol_bar_size = 20
tick_bar_size = 20
dollar_bar_size = 400000

plot_morning_time = date + "_morning_" + time_sampling_rule + "_time"
plot_afternoon_time = date + "_afternoon_" + time_sampling_rule + "_time"
plot_midnight_time = date + "_midnight_" + time_sampling_rule + "_time"

plot_morning_vol = date + "_morning_" + str(vol_bar_size) + "_volume"
plot_afternoon_vol = date + "_afternoon_" + str(vol_bar_size) + "_volume"
plot_midnight_vol = date + "_midnight_" + str(vol_bar_size) + "_volume"

plot_morning_tick = date + "_morning_" + str(tick_bar_size) + "_tick"
plot_afternoon_tick = date + "_afternoon_" + str(tick_bar_size) + "_tick"
plot_midnight_tick = date + "_midnight_" + str(tick_bar_size) + "_tick"

plot_morning_dollar = date + "_morning_" + str(dollar_bar_size) + "_dollar"
plot_afternoon_dollar = date + "_afternoon_" + str(dollar_bar_size) + "_dollar"
plot_midnight_dollar = date + "_midnight_" + str(dollar_bar_size) + "_dollar"

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

image_folders_path = os.path.join(root_dir, "data_plots", "raw_bar_data_fine")

win_x_inches = 16
win_y_inches = 8

image_folder_path = os.path.join(image_folders_path, date)
if not os.path.isdir(image_folder_path):
    os.mkdir(image_folder_path)

time_morning_plot_path = os.path.join(image_folder_path, "time_morning.png")
time_afternoon_plot_path = os.path.join(image_folder_path, "time_afternoon.png")
time_midnight_plot_path = os.path.join(image_folder_path, "time_midnight.png")

volume_morning_plot_path = os.path.join(image_folder_path, "volume_morning.png")
volume_afternoon_plot_path = os.path.join(image_folder_path, "volume_afternoon.png")
volume_midnight_plot_path = os.path.join(image_folder_path, "volume_midnight.png")

tick_morning_plot_path = os.path.join(image_folder_path, "tick_morning.png")
tick_afternoon_plot_path = os.path.join(image_folder_path, "tick_afternoon.png")
tick_midnight_plot_path = os.path.join(image_folder_path, "tick_midnight.png")

dollar_morning_plot_path = os.path.join(image_folder_path, "dollar_morning.png")
dollar_afternoon_plot_path = os.path.join(image_folder_path, "dollar_afternoon.png")
dollar_midnight_plot_path = os.path.join(image_folder_path, "dollar_midnight.png")


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


# ------ plotting time sampled -----------
morning_time_VVAP = time_morning_df["VVAP"].mask(lambda x : x == 0).interpolate()
afternoon_time_VVAP = time_afternoon_df["VVAP"].mask(lambda x : x == 0).interpolate()
midnight_time_VVAP = time_midnight_df["VVAP"].mask(lambda x : x == 0).interpolate()

plt.figure()
pltr.plot_xy_nano_time(time_morning_df["timestamp"].to_numpy(), morning_time_VVAP, y_label = "VVAP", title = plot_morning_time)
plt.grid(linewidth = 1)
pltr.save_plot(time_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_nano_time(time_afternoon_df["timestamp"].to_numpy(), afternoon_time_VVAP, y_label = "VVAP", title = plot_afternoon_time)
plt.grid(linewidth = 1)
pltr.save_plot(time_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy_nano_time(time_midnight_df["timestamp"].to_numpy(), midnight_time_VVAP, y_label = "VVAP", title = plot_midnight_time)
plt.grid(linewidth = 1)
pltr.save_plot(time_midnight_plot_path, win_x_inches, win_y_inches)

# --------- plotting volume sampled -----------
plt.figure()
pltr.plot_xy(volume_morning_df["volume_time"], volume_morning_df["VVAP"], x_label = "volume traded", y_label = "VVAP", title = plot_morning_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(volume_afternoon_df["volume_time"], volume_afternoon_df["VVAP"], x_label = "volume traded", y_label = "VVAP", title = plot_afternoon_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(volume_midnight_df["volume_time"], volume_midnight_df["VVAP"], x_label = "volume traded", y_label = "VVAP", title = plot_midnight_vol)
plt.grid(linewidth = 1)
pltr.save_plot(volume_midnight_plot_path, win_x_inches, win_y_inches)

# ----- plotting tick sampled -------
plt.figure()
pltr.plot_xy(tick_morning_df["tick_time"], tick_morning_df["VVAP"], x_label = "ticks passed", y_label = "VVAP", title = plot_morning_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(tick_afternoon_df["tick_time"], tick_afternoon_df["VVAP"], x_label = "ticks passed", y_label = "VVAP", title = plot_afternoon_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(tick_midnight_df["tick_time"], tick_midnight_df["VVAP"], x_label = "ticks passed", y_label = "VVAP", title = plot_midnight_tick)
plt.grid(linewidth = 1)
pltr.save_plot(tick_midnight_plot_path, win_x_inches, win_y_inches)

# ------ plotting dollar sampled -----------
plt.figure()
pltr.plot_xy(dollar_morning_df["dollar_time"], dollar_bar_size / dollar_morning_df["volume"], x_label = "dollars traded"
             , y_label = "VVAP", title = plot_morning_dollar)
plt.grid(linewidth = 1)
pltr.save_plot(dollar_morning_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(dollar_afternoon_df["dollar_time"], dollar_bar_size / dollar_afternoon_df["volume"], x_label = "dollars traded"
             , y_label = "VVAP", title = plot_afternoon_dollar)
plt.grid(linewidth = 1)
pltr.save_plot(dollar_afternoon_plot_path, win_x_inches, win_y_inches)

plt.figure()
pltr.plot_xy(dollar_midnight_df["dollar_time"], dollar_bar_size / dollar_midnight_df["volume"], x_label = "dollars traded"
             , y_label = "VVAP", title = plot_midnight_dollar)
plt.grid(linewidth = 1)
pltr.save_plot(dollar_midnight_plot_path, win_x_inches, win_y_inches)

# ------ END -----