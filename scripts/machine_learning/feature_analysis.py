import definitions
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import plotter.plotter as pltr
from matplotlib.ticker import MaxNLocator

# ----- load data -----
root_dir = definitions.ROOTDIR
bar_data_folder_path = os.path.join(root_dir, "data_bar")
date_str = "20170307"
period_str = "morning"
sampling_type_str = "volume"
bar_size_str = "100"
name_str = date_str + "_" + period_str + "_" + sampling_type_str + "_" + bar_size_str
bar_file_path = os.path.join(bar_data_folder_path, "volume_sampled", name_str + "_sampled.csv")
main_plot_folder_path = os.path.join(root_dir, "data_plots", "feature_set_1")

plot_folder_path = os.path.join(main_plot_folder_path, date_str)
if not os.path.isdir(plot_folder_path):
    os.mkdir(plot_folder_path)
plot_folder_path = os.path.join(plot_folder_path, period_str)
if not os.path.isdir(plot_folder_path):
    os.mkdir(plot_folder_path)

win_x_inches = 16
win_y_inches = 8

atr_time_period = 15
adx_time_period = 15
apo_fast_period = 4
apo_slow_period = 12
rsi_time_period = 30
roc_time_period = 30
ult_osc_period_1 = 5
ult_osc_period_2 = 10
ult_osc_period_3 = 15
tsf_time_period = 15
p_corr_time_period = 15
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9

bar_df = pd.read_csv(bar_file_path)
bar_df = fe.create_feature_set_1(bar_df, atr_time_period = atr_time_period, adx_time_period = adx_time_period, apo_fast_period = apo_fast_period,
    rsi_time_period = rsi_time_period, roc_time_period = roc_time_period, ult_osc_period_1 = ult_osc_period_1, ult_osc_period_2 = ult_osc_period_2,
    tsf_time_period = tsf_time_period, p_corr_time_period = p_corr_time_period , macd_fast_period = macd_fast_period , macd_slow_period = macd_slow_period,
    macd_signal_period = macd_signal_period )
ta_indicators = ["atr", "adx", "apo", "rsi", "roc", "ult_osc", "log_tsf", "per_corr", "macd", "macd_signal", "macd_hist"]

ta_ind_descrp = {
    "atr" : str(atr_time_period),
    "adx" : str(adx_time_period),
    "apo" : str(apo_fast_period) + "_" + str(apo_slow_period),
    "rsi" : str(rsi_time_period),
    "roc" : str(roc_time_period),
    "ult_osc" : str(ult_osc_period_1) + "_" + str(ult_osc_period_2) + " " + str(ult_osc_period_3),
    "log_tsf" : str(tsf_time_period),
    "per_corr" : str(p_corr_time_period),
    "macd" : str(macd_fast_period) + "_" + str(macd_slow_period),
    "macd_signal" : str(macd_fast_period) + "_" + str(macd_slow_period),
    "macd_hist" : str(macd_fast_period) + "_" + str(macd_slow_period)
}

bar_df.dropna(inplace = True)

# ----- plotting ------

for ta_ind in ta_indicators:
    fig, ax = plt.subplots(2)
    fig.suptitle(name_str + "_" + ta_ind + "_" + ta_ind_descrp[ta_ind])
    ax[0].xaxis.set_major_locator(MaxNLocator(12))
    ax[1].xaxis.set_major_locator(MaxNLocator(12))
    ax[0].set_title("VVAP")
    ax[1].set_title(ta_ind)
    ax[1].set_xlabel("volume traded")
    ax[0].set_ylabel("price")
    ax[1].set_ylabel(ta_ind)

    ax[0].plot(bar_df["volume_time"], bar_df["VVAP"])
    ax[1].plot(bar_df["volume_time"], bar_df[ta_ind])
    ax[0].grid(linewidth = 1)
    ax[1].grid(linewidth = 1)
    plot_file_path = os.path.join(plot_folder_path, ta_ind +"_" + ta_ind_descrp[ta_ind] + ".png")
    pltr.save_plot(plot_file_path, win_x_inches, win_y_inches)


# ---- correlation between features ------
features = bar_df[ta_indicators]
sns.heatmap(features.corr(), vmin=-1, vmax=1, annot=True)
plot_file_path = os.path.join(plot_folder_path, "correlation_matrix.png")
pltr.save_plot(plot_file_path, win_x_inches, win_y_inches)

# ---- stationary tests -------
print("# ------------- #")
for ta_ind in ta_indicators:
    feature = bar_df[ta_ind]
    result = adfuller(feature)
    print("Feature : " + ta_ind)
    print(result)
    print("# ------------- #")

