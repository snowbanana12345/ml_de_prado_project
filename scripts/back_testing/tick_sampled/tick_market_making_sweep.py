from back_testing.limit_order_strategies import MarketMaker
from back_testing.post_trade import get_limit_strategy_statistics
from data_base.data_base import instance as db
import plotter.plotter as pltr
import plotter.back_test_plotting as bpltr
import matplotlib.pyplot as plt

"""
Here we are going to be a market maker
We set our bid limit order to be at the opening bid 
and our ask limit order to be at the opening ask
To deal with inventory imbalances, for every few excess positions, we increase our limit order to discourage further imbalance
"""

# ------- custom inputs ------
bar_size = "100"
period = "morning"
sampling = "tick"
date_lst = ["20170125", "20170126", "20170127", "20170131", "20170201", "20170202", "20170203", "20170206", "20170207"
            , "20170208",  "20170209",  "20170210",  "20170213", "20170214", "20170215", "20170216", "20170217", "20170220"]

threshold = 1
tick_size = 5
time_col = "tick_time"
result_lst = []
trader = MarketMaker(threshold=threshold, tick_size=tick_size, ask_reference = "open", bid_reference = "open",
                     ask_trigger = "high", bid_trigger = "low", equals = True, ignore_equality = True, min_spread = 2)
# ----- loop ----
for date in date_lst:
    # ----- load data ----
    bar_df = db.get_tick_bar_wlimit(date, period, bar_size)
    # ----- trade -----
    trade_df = trader.trade(bar_df, time_col)
    result = get_limit_strategy_statistics(trade_df)
    result_lst.append(result)

# ---- print out final statistics ----
for date, result in zip(date_lst, result_lst):
    print(date, result)