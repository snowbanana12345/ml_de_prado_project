import definitions
import os
import pickle
import pandas as pd
import numpy as np
import plotter.plotter as pltr
import plotter.back_test_plotting as bpltr
import matplotlib.pyplot as plt

"""
Strategy : 
if upper_rfr predicts 1, the price will break the upper barrier, we hold a position of 1
if lower_rfr predicts 1, the price will break the lower barrier, we hold a position of -1
if both predicts 1, we cannot trust the models, we hold a position of 0
Basically, a logical XOR.

model is only used at the valid points in times determined by the filter
Assume that we trade at the close of the bar we make the prediction, does not account for slippage
No transaction cost, the transaction cost is estimated using (no of trades) * (average bid,ask spread)
Close position at the end of period, there will be at least look_ahead number of bars to do so

We also use market orders, i.e buy at the closing ask, sell at the closing bid
"""
# ------- custom inputs ------
root_dir = definitions.ROOTDIR
sampling = "volume"
sampling_size = "100"
data_date = "20170303"
time_period = "morning"
bar_data_folder = os.path.join(root_dir, "data_bar_wlimit", sampling + "_sampled")
bar_data_file_path = os.path.join(bar_data_folder, data_date + "_" + time_period + "_" + sampling + "_" + sampling_size + "_sampled.csv")

models_folder = os.path.join(root_dir, "trained_ml_models", "rfr_double_barrier")
model_file_path = os.path.join(models_folder, "coarse_15_5_5.pickle")

# ------ read data and predict -----
bar_df = pd.read_csv(bar_data_file_path)
pickle_in = open(model_file_path, "rb")
model = pickle.load(pickle_in)
pickle_in.close()
bar_pred_df = model.predict(bar_df)
look_ahead, upper_barrier, lower_barrier = model.get_params()

# ------- trade_function ---------
def trade_func(upper : int, lower : int):
    if upper == 0 and lower == 0:
        return 0
    elif upper == 1 and lower == 0:
        return 1
    elif upper == 0 and lower == 1:
        return -1
    else :
        return 0

def correct_pnl(position : int, pnl):
    if position == 0:
        return pnl
    else :
        return np.nan

def market_order(trade, bid, ask):
    if trade < 0:
        return - trade * bid
    elif trade > 0:
        return -trade * ask
    else:
        return 0


# ----- trade -------
bar_pred_df["positions"] = bar_pred_df.apply(lambda row : trade_func(upper = row["upper_barrier_prediction"]
                                                                     , lower = row["lower_barrier_prediction"]), axis = 1)
bar_df["positions"] = bar_pred_df["positions"]
bar_df["positions"] = bar_df["positions"].ffill()
bar_df = bar_df.fillna(0)
bar_df.loc[len(bar_df) - look_ahead : len(bar_df), "positions"] = 0
bar_df["trades"] = bar_df["positions"].diff().bfill()
bar_df["trade_amounts"] = bar_df.apply(lambda row : market_order(row["trades"], row["close_bid"], row["close_ask"]), axis = 1)
bar_df["PnL"] = bar_df["trade_amounts"].cumsum()
bar_df["PnL"] = bar_df.apply(lambda row : correct_pnl(position = row["positions"], pnl = row["PnL"]), axis = 1)
bar_df["PnL"] = bar_df["PnL"].ffill()

# ------ plot results -----
plt.figure(0)
pltr.plot_xy(bar_df["volume_time"], bar_df["PnL"], title = "Realized PnL", x_label = "volume time", y_label = "Profits")

plt.figure(1)
bpltr.plot_trading_mo(bar_df, "volume_time", "VVAP", "trades", x_label ="volume_time", y_label ="VVAP", title ="Trades")
plt.show()


