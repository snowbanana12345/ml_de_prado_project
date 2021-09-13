from back_testing.back_test import TradeSimulator
from back_testing.limit_order_strategies import MarketMakerSimulator
import back_testing.post_trade as pt
from data_base.data_base import instance as db
import math
import definitions
import time


# ------- custom inputs ------
bar_size = "100"
period = "afternoon"
sampling = "tick"
date_lst = ["20170125", "20170126", "20170127", "20170131", "20170201", "20170202", "20170203", "20170206", "20170207"
            , "20170208",  "20170209",  "20170210",  "20170213", "20170214", "20170215", "20170216", "20170217", "20170220"]
ask_reference = definitions.OPEN
bid_reference = definitions.OPEN
threshold = 1
tick_size = 5
time_col = definitions.TICK_TIME
position_max_duration = 10
limit_order_standing = 10
take_profit = 5
stop_loss = math.inf
size = 1

# ----- initialize objects -----
trade_sim = TradeSimulator()
mm_sim = MarketMakerSimulator(trade_simulator = trade_sim, ask_reference = ask_reference, bid_reference = bid_reference,
                              threshold = threshold, tick_size = tick_size, position_max_duration = position_max_duration,
                              limit_order_standing = limit_order_standing, take_profit = take_profit, stop_loss = stop_loss,
                              size = size)
# ----- trade ----
result_lst = []
for date in date_lst:
    start_time = time.time()
    mm_sim.reset()
    bar_df = db.get_bar_wlimit(date, period, bar_size, sampling)
    # ----- trade ------
    mm_sim.trade(bar_df)

    trade_df = mm_sim.get_results()
    bar_df = pt.process_trade_df(bar_df, trade_df)
    result = pt.get_trade_result(bar_df, trade_df)
    result_lst.append(result)
    end_time = time.time()
    print("Trading complete : time elapsed : " + str(end_time - start_time))

# ---- print out results -----
for date,result in zip(date_lst,result_lst):
    print(date, result)