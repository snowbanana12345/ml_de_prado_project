from scipy import stats
import numpy as np
import pandas as pd
from datetime import datetime

def print_diagnostic(df, price_z_score_threshold = 10, time_z_score_threshold = 3):
    """ df is completely unprocessed data frame, assuming the data has the format
    timestampNano,lastPrice,lastQty,ask1p,ask1q,ask2p,ask2q,ask3p,ask3q,ask4p,ask4q,ask5p
    ,ask5q,bid1p,bid1q,bid2p,bid2q,bid3p,bid3q,bid4p,bid4q,bid5p,bid5q"""
    print(df.info())
    print_has_time_stamp_duplicates(df)
    scan_missing_values(df)
    print_time_interval_stats(df, time_z_score_threshold)
    outlier_scanning(df, price_z_score_threshold)
    scan_invalid_ask_order(df)
    scan_invalid_bid_order(df)
    scan_bid_higher_than_ask(df)
    scan_zero_or_negative_quantity(df)
    scan_valid_price_and_quantity(df)

    print_limit_quantity_stats(df)
    print_last_trade_quantity_stats(df)

def time_stamp_comparison(time_stamp_series):
    pd_dt = time_stamp_series.apply(pd.Timestamp)
    def nano_time_to_date_time(time_stamp_nano):
        return datetime.fromtimestamp(time_stamp_nano // 1000000000)
    dt_dt = time_stamp_series.apply(nano_time_to_date_time)
    print(" -------- Pandas time stamp conversion ------")
    print(pd_dt.head(10))
    print(" -------- Python built in time stamp conversion -------")
    print(dt_dt.head(10))

def print_limit_quantity_stats(df):
    # -------- limit quantity statistics --------------
    print(" --------- Quantity statistics ---------- ")
    qty_symbols = ["ask1q", "ask2q", "ask3q", "ask4q", "ask5q", "bid1q", "bid2q", "bid3q", "bid4q", "bid5q"]
    for symbol in qty_symbols:
        print("Quantity : " + symbol + " mean : " + str(df[symbol].mean()) + " and std : " + str(df[symbol].std()))


def print_last_trade_quantity_stats(df, last_qty_col = "lastQty"):
    last_qty = df[last_qty_col]
    non_zero_last_qty = last_qty[last_qty > 0]
    print("Quantity : lastQty mean : " + str(non_zero_last_qty.mean()) + " and std : " + str(non_zero_last_qty.std())
          +" total qty : " + str(non_zero_last_qty.sum()))


def scan_valid_price_and_quantity(df, price_col = "lastPrice", quantity_col = "lastQty"):
    def valid_last_price_qty_pair(lp, lq):
        lp_invalid = lp <= 0 or pd.isnull(lp)
        lq_invalid = lq <= 0 or pd.isnull(lq)
        return lp_invalid != lq_invalid
    invalid_lastprice_qty_pairs = df.apply(lambda x: valid_last_price_qty_pair(x[price_col], x[quantity_col]), axis=1)
    num_invalid = len(invalid_lastprice_qty_pairs[invalid_lastprice_qty_pairs])
    print("Number of invalid last price and lats qty pairs : " + str(num_invalid))


def scan_zero_or_negative_quantity(df):
    qty_symbols = ["ask1q", "ask2q", "ask3q", "ask4q", "ask5q", "bid1q", "bid2q", "bid3q", "bid4q", "bid5q"]
    def any_zero_or_negative(*args):
        truths = False
        for i in range(len(args) - 1):
            truths = np.logical_or(truths, args[i] <= 0)
        return truths
    wrong_qtys = df.apply(lambda x: any_zero_or_negative(*[x[symbol] for symbol in qty_symbols]), axis=1)
    num_wrong_qtys = len(wrong_qtys[wrong_qtys])
    print("Number of zero or negative limit order quantities : " + str(num_wrong_qtys))


def scan_bid_higher_than_ask(df):
    bid_larger_than_ask = df.apply(lambda x: np.greater(x["bid1p"], x["ask1p"]), axis=1)
    num_bid_larger_than_ask = len(bid_larger_than_ask[bid_larger_than_ask])
    print("Number of best bids larger than the best ask : " + str(num_bid_larger_than_ask))


def scan_invalid_bid_order(df):
    def wrong_bid_order(*args):
        truths = True
        for i in range(len(args) - 1):
            truths = np.logical_and(truths, np.greater(args[i], args[i + 1]))
        return np.logical_not(truths)

    bid_limit_symbols = ["bid1p", "bid2p", "bid3p", "bid4p", "bid5p"]
    wrongly_ordered = df.apply(lambda x: wrong_bid_order(*[x[bid_col] for bid_col in bid_limit_symbols]), axis=1)
    num_wrongly_ordered = len(wrongly_ordered[wrongly_ordered])
    print("Number of bid rows that are wrongly ordered : " + str(num_wrongly_ordered))


def scan_invalid_ask_order(df):
    print("------- Scanning for analomous patterns in the bid ask order book ---------")

    def wrong_ask_order(*args):
        truths = True
        for i in range(len(args) - 1):
            truths = np.logical_and(truths, np.greater(args[i + 1], args[i]))
        return np.logical_not(truths)

    ask_limit_symbols = ["ask1p", "ask2p", "ask3p", "ask4p", "ask5p"]
    wrongly_ordered = df.apply(lambda x: wrong_ask_order(*[x[ask_col] for ask_col in ask_limit_symbols]), axis=1)
    num_wrongly_ordered = len(wrongly_ordered[wrongly_ordered])
    print("Number of ask rows that are wrongly ordered : " + str(num_wrongly_ordered))


def outlier_scanning(df, price_z_score_threshold, limit_symbols = ("ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p")):
    # ------ outlier scanning --------
    print(" ---------- Scanning for outliers ------------- ")
    for symbol in limit_symbols:
        z_scores = np.abs(stats.zscore(df[symbol]))
        outliers = z_scores[z_scores > price_z_score_threshold]
        print("Data in symbol : " + symbol + " has " + str(len(outliers)) + " outliers with z score > " + str(
            price_z_score_threshold))


def print_time_interval_stats(df, time_z_score_threshold):
    # ------ time interval statistics --------
    print("---------- Time interval statistics -----------")
    time_dff = df.timestampNano.diff()
    print("Mean time interval : " + str(time_dff.mean() / 1000000) + " ms")
    print("Time interval standard deviation : " + str(time_dff.std() / 1000000) + " ms")
    z_scores = np.abs(stats.zscore(time_dff))
    outliers = z_scores[z_scores > time_z_score_threshold]
    print("Number of time interval outliers : " + str(len(outliers)) + " with z score > " + str(time_z_score_threshold))


def scan_missing_values(df):
    # ------ missing value scanning -------
    print(" ---------- Scanning for missing values ---------------")
    for col in df.columns:
        colm = df[col]
        num_missing_values = len(colm[pd.isnull(colm)])
        print("Column : " + col + " has " + str(num_missing_values) + " missing values")

def print_has_time_stamp_duplicates(df):
    # ------ time stamp duplicate scanning -------
    print("Are there time stamp duplicates? : " + str(df.timestampNano.duplicated().any()))

def scan_zeros(df):
    print(" ---------- Scanning for zeros ---------------")
    print("data frame has total of " + str(len(df)) + " rows")
    for col in df.columns:
        colm = df[col]
        num_zeros = len(colm[colm == 0])
        print("Column : " + col + " has " + str(num_zeros) + " zero values")

def print_ochl(tick_df, row_name):
    open_price = tick_df[row_name].values[0]
    close_price = tick_df[row_name].values[-1]
    high = tick_df[row_name].max()
    low = tick_df[row_name].min()
    print(open_price)
    print(close_price)
    print(high)
    print(low)