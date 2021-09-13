import matplotlib.pyplot as plt
from datetime import datetime
import math


def plot_trade_volume(trade_qty_series, title, rule = "600S"):
    plot_title = title + " : Total trade volume in intervals of : " + rule
    trade_qty_series.resample(rule).sum().plot(title = plot_title, label = "last quantity")

def plot_bid_ask_spread(bid_series, ask_series, title, rule = "600S"):
    spread_series = ask_series - bid_series
    plot_title = title + " : Average bid-ask spread in intervals of : " + rule
    spread_series.resample(rule).mean().plot(title = plot_title, label = "bid/ask spread")

def plot_bid_ask_volatility(bid_series, ask_series, title, rule = "600S"):
    mid_point_series = (bid_series + ask_series) / 2
    plot_title = title + " : volatility of the mid price point in intervals of : " + rule
    mid_point_series.resample(rule).std().plot(title =plot_title, label = "volatility")

def plot_trade_price_volatility(trade_qty_series, title, rule = "600S"):
    plot_title = title + " : volatility of the last trade price in intervals of : " + rule
    trade_qty_series[trade_qty_series > 0].resample(rule).std().plot(title = plot_title, label = "volatility")

def plot_log_return(bid_series, ask_series, title):
    mid_point_series = (bid_series + ask_series) / 2
    plot_title = title + " : log returns "
    mid_point_series.apply(math.log).diff().dropna().plot(title = plot_title, label = " log returns")

def plot_log_return_resampled(bid_series, ask_series, title, rule = "60S"):
    mid_point_series = (bid_series + ask_series) / 2
    plot_title = title + " : log returns in intervals of : " + rule
    mid_point_series.apply(math.log).diff().dropna().resample(rule).sum().plot(title = plot_title, label = " log returns")

def plot_time_sampled(df, title, cols = ("open", "close", "high", "low", "total_volume", "VVAP")):
    for i, col in enumerate(cols):
        plot_title = title + " : Time sampled bars : " + col
        plt.figure(i + 1)
        if col == "total_volume":
            df[col].plot(title = plot_title)
        else :
            column = df[col]
            column = column[column > 0]
            column.plot(title = plot_title)
