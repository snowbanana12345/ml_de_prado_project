from back_testing.limit_order_strategies import MarketMaker
from back_testing.post_trade import get_limit_strategy_statistics
from data_base.data_base import instance as db
import plotter.plotter as pltr
import plotter.back_test_plotting as bpltr
import matplotlib.pyplot as plt

# ------- custom inputs ------
bar_size = "100"
period = "morning"
date = "20170125"

threshold = 1
tick_size = 5
time_col = "tick_time"
result_lst = []
trader_1 = MarketMaker(threshold=threshold, tick_size=tick_size, ask_reference = "open_ask", bid_reference = "open_bid",
                     ask_trigger = "high_ask", bid_trigger = "low_bid", equals = True, ignore_equality = True, min_spread = 2)
#trader_2 = MarketMaker(threshold=threshold, tick_size=tick_size, ask_reference = "open_ask", bid_reference = "open_bid",
                   #  ask_trigger = "high", bid_trigger = "low", equals = True)
bar_df = db.get_tick_bar_wlimit(date, period, bar_size)
trade_df_1 = trader_1.trade(bar_df, time_col)
result = get_limit_strategy_statistics(trade_df_1)
print(result)
#trade_df_2 = trader_2.trade(bar_df, time_col)

bpltr.plot_trade_limit(trade_df_1, time_col, "price_series", "buys", "buy_price", "sells", "sell_price", "end_positions",
                      "Pnl")
#bpltr.plot_trade_limit(trade_df_2, time_col, "price_series", "buys", "buy_price", "sells", "sell_price", "end_positions",
                       #"Pnl")
plt.show()