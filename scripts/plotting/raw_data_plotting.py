import plotter.stats_plotter as stat_plt
import plotter.plotter as pltr
import os
import definitions
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing.data_cleaning as dat_clean

# ----------- user inputs ----------------
root_dir = definitions.ROOTDIR
date = "20170307"
data_file_path = os.path.join(root_dir, "data_raw_tick", "ModelDepthProto_" + date + ".csv")
image_folders_path = os.path.join(root_dir, "data_plots", "raw_tick_data")

win_x_inches = 16
win_y_inches = 8

image_folder_path = os.path.join(image_folders_path, date)
if not os.path.isdir(image_folder_path):
    os.mkdir(image_folder_path)

raw_data_path = os.path.join(image_folder_path, "raw_data.png")
trade_volume_path = os.path.join(image_folder_path, "trade_volume.png")
bid_ask_spread_path = os.path.join(image_folder_path, "bid_ask_spread.png")
trade_price_volatility_path = os.path.join(image_folder_path, "trade_price_volatility.png")
bid_ask_volatility_path = os.path.join(image_folder_path, "bid_ask_volatility.png")
log_return_path =  os.path.join(image_folder_path, "log_return.png")


# ----- data diagnostics -----
tick_df = pd.read_csv(data_file_path)
#dat_diag.print_diagnostic(tick_df)
tick_df = dat_clean.pre_process(tick_df)

# ----- plot statistics ------
plt.figure()
pltr.plot_xy_nano_time(tick_df["timestampNano"].to_numpy(), tick_df["ask1p"], y_label = ["ask1p"], title = date)
pltr.save_plot(raw_data_path, win_x_inches, win_y_inches)

plt.figure()
stat_plt.plot_trade_volume(tick_df["lastQty"], date, rule ="300S")
plt.gcf().set_size_inches(win_x_inches, win_y_inches)
plt.savefig(trade_volume_path,  dpi = 100)
plt.close()

plt.figure()
stat_plt.plot_bid_ask_spread(tick_df["bid1p"], tick_df["ask1p"], date, rule ="300S")
plt.gcf().set_size_inches(win_x_inches, win_y_inches)
plt.savefig(bid_ask_spread_path,  dpi = 100)
plt.close()

plt.figure()
stat_plt.plot_trade_price_volatility(tick_df["lastPrice"], date, rule ="900S")
plt.gcf().set_size_inches(win_x_inches, win_y_inches)
plt.savefig(trade_price_volatility_path,  dpi = 100)
plt.close()

plt.figure()
stat_plt.plot_bid_ask_volatility(tick_df["bid1p"], tick_df["ask1p"], date, rule ="900S")
plt.gcf().set_size_inches(win_x_inches, win_y_inches)
plt.savefig(bid_ask_volatility_path,  dpi = 100)
plt.close()

plt.figure()
stat_plt.plot_log_return_resampled(tick_df["bid1p"], tick_df["ask1p"], date, rule = "300S")
plt.gcf().set_size_inches(win_x_inches, win_y_inches)
plt.savefig(log_return_path,  dpi = 100)
plt.close()

# plt.legend()
# plt.show()