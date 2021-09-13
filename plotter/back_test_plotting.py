import pandas as pd
import plotter.plotter as pltr
import numpy as np
import matplotlib.pyplot as plt

def nano_time_to_date_time(ts):
    return (pd.Timestamp(ts) + pd.to_timedelta(8, unit = "h"))

def plot_back_test(trade_record_df, title):
    """ trade_record_df contains columns : timestamp,positions,trade_qty,trade_amount,realized """
    # time = trade_record_df["timestamp"].apply(nano_time_to_date_time).apply(lambda x: x.time())
    pltr.plot_xy_nano_time(np.array(trade_record_df["timestamp"]), trade_record_df["realized"], y_label = "realized_PnL",
                          num_x_ticks = 10, title = title + "_realized_PnL")

def plot_trading_mo(df : pd.DataFrame, time_series_col : str, price_series_col : str, trade_series_col : str
                    , x_label ="time", y_label ="price", title ="plot")->None:
    buys = df[df[trade_series_col] > 0]
    sells = df[df[trade_series_col] < 0]
    plt.plot(df[time_series_col], df[price_series_col], label = "price_series")
    plt.scatter(buys[time_series_col], buys[price_series_col], label = "buy", color = "green", marker = "^")
    plt.scatter(sells[time_series_col], sells[price_series_col], label = "sell", color = "red", marker = "v")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def plot_trade_limit(trade_df : pd.DataFrame, time_col : str, price_series_col : str, buy_series_col : str,
                     buy_prices_col : str, sell_series_col : str, sell_prices_col : str, end_positions_col : str
                     ,pnl_col : str):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("trading")
    ax1.plot(trade_df[time_col], trade_df[price_series_col], label="price_series")
    ax1.scatter(trade_df[time_col][trade_df[buy_series_col] > 0], trade_df[buy_prices_col][trade_df[buy_series_col] > 0]
                , color="green", marker="^")
    ax1.scatter(trade_df[time_col][trade_df[sell_series_col] > 0], trade_df[sell_prices_col][trade_df[sell_series_col] > 0]
                , color="red", marker="v")
    pnl_series = trade_df.loc[trade_df[end_positions_col] == 0, pnl_col]
    time_series = trade_df.loc[trade_df[end_positions_col] == 0, time_col]

    ax2.plot(time_series, pnl_series, label="pnl")
    ax2.set_xlabel(time_col)
    ax1.grid(linewidth=1)
    ax2.grid(linewidth=1)
    ax1.legend()
    ax2.legend()



