from back_testing.back_test import TradeSimulator
from back_testing.limit_order_strategies import MarketMakerSimulator
import back_testing.post_trade as pt
from data_base.data_base import instance as db
import matplotlib.pyplot as plt
import math
import definitions


# ------- custom inputs ------
bar_size = "100"
period = "afternoon"
date = "20170131"
sampling = "tick"

ask_reference = definitions.OPEN
bid_reference = definitions.OPEN
threshold = 1
tick_size = 5
time_col = definitions.TICK_TIME
position_max_duration = 5
limit_order_standing = 3
take_profit = 5
stop_loss = math.inf
size = 1

# ----- initialize objects -----
bar_df = db.get_tick_bar_wlimit(date, period, bar_size)
trade_sim = TradeSimulator()
mm_sim = MarketMakerSimulator(trade_simulator = trade_sim, ask_reference = ask_reference, bid_reference = bid_reference,
                              threshold = threshold, tick_size = tick_size, position_max_duration = position_max_duration,
                              limit_order_standing = limit_order_standing, take_profit = take_profit, stop_loss = stop_loss,
                              size = size)
# ----- trade ------
mm_sim.trade(bar_df)
trade_df = mm_sim.get_results()
bar_df = pt.process_trade_df(bar_df, trade_df)
result = pt.get_trade_result(bar_df, trade_df)
print(result)
# ------ plot pnl ------
fig, (ax1, ax2) = plt.subplots(2,1)
fig.suptitle(date + " " + period)
ax1.plot(bar_df[time_col], bar_df[definitions.VWAP])
ax1.set_ylabel("VWAP")
ax2.plot(bar_df[time_col], bar_df["pnl"])
ax2.set_ylabel("pnl")
ax2.set_xlabel("volume traded")
plt.show()