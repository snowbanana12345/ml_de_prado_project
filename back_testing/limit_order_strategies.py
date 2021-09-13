import pandas as pd
import numpy as np
import time
import definitions
from back_testing.back_test import TradeSimulator


class MarketMaker:
    def __init__(self, threshold : int, tick_size : int, ask_reference : str
                 , bid_reference : str, ask_trigger : str, bid_trigger : str,
                 equals : bool, ignore_equality : bool, min_spread : int):
        self.threshold = threshold
        self.tick_size = tick_size
        self.ask_reference = ask_reference
        self.bid_reference = bid_reference
        self.ask_trigger = ask_trigger
        self.bid_trigger = bid_trigger
        self.equals = equals
        self.ignore_equality = ignore_equality
        self.min_spread = min_spread

    def trade(self, bar_df : pd.DataFrame, time_col : str) -> pd.DataFrame:
        """ NOTE : assumes bar_df is range indexed
            NOTE : settings equals to True and ask_trigger to be the ask will simply result in buying at the bid and selling
            and the ask on every bar which is nonsense.
            NOTE : setting the bid/ask reference to close causes look ahead bias
        """
        for col in ["open", "close", "high", "low", "VVAP", "open_ask", "close_ask", "high_ask", "low_ask",
                    "close_ask", "open_bid", "close_bid", "high_bid", "low_bid"]:
            if col not in bar_df.columns:
                raise ValueError(col + " : not available in input bar data")
        start_time = time.time()
        # ----- START -----
        positions = np.zeros(len(bar_df)).astype('int32')
        buys = np.zeros(len(bar_df)).astype('int32')
        buy_prices = np.zeros(len(bar_df)).astype('int32')
        offered_buy_prices = np.zeros(len(bar_df)).astype('int32')
        sells = np.zeros(len(bar_df)).astype('int32')
        sell_prices = np.zeros(len(bar_df)).astype('int32')
        offered_sell_prices = np.zeros(len(bar_df)).astype('int32')
        for i, row in bar_df.iterrows():
            if i >= len(bar_df) - 1:
                break
            imbalance = int(positions[i] / self.threshold)
            ask_ref = max(row[self.ask_reference], row["open_bid"] + self.tick_size)
            bid_ref = min(row[self.bid_reference], row["open_ask"] - self.tick_size)
            # raise the ask price when we have net negative position
            ask_price = ask_ref if imbalance >= 0 else ask_ref + abs(imbalance) * self.tick_size
            # lower the bid price when we have net positive position
            bid_price = bid_ref if imbalance <= 0 else bid_ref - abs(imbalance) * self.tick_size
            if ask_price == bid_price and not self.ignore_equality:
                ask_price += self.min_spread * self.tick_size
                bid_price -= self.min_spread * self.tick_size
            if self.equals:
                if ask_price <= row[self.ask_trigger]:
                    sells[i] = 1
                    sell_prices[i] = ask_price
                if bid_price >= row[self.bid_trigger]:
                    buys[i] = 1
                    buy_prices[i] = bid_price
            else :
                if ask_price < row[self.ask_trigger]:
                    sells[i] = 1
                    sell_prices[i] = ask_price
                if bid_price > row[self.bid_trigger]:
                    buys[i] = 1
                    buy_prices[i] = bid_price
            offered_buy_prices[i] = bid_price
            offered_sell_prices[i] = ask_price
            positions[i + 1] = positions[i] + buys[i] - sells[i]

        # ---- close last position -----
        if positions[-1] < 0:
            buys[-1] = -positions[-1]
            buy_prices[-1] = bar_df.loc[len(bar_df) - 1, "close_ask"]
        if positions[-1] > 0:
            sells[-1] = positions[-1]
            sell_prices[-1] = bar_df.loc[len(bar_df) - 1, "close_bid"]
        offered_buy_prices[-1] = bar_df.loc[len(bar_df) - 1, "close_bid"]
        offered_sell_prices[-1] = bar_df.loc[len(bar_df) - 1, "close_ask"]

        # ---- find trade statistics -----
        trade_df = pd.DataFrame({
            time_col: bar_df[time_col],
            "price_series": bar_df["VVAP"],
            "positions": positions,
            "buys": buys,
            "buy_price": buy_prices,
            "offered_bid": offered_buy_prices,
            "sells": sells,
            "sell_price": sell_prices,
            "offered_ask": offered_sell_prices
        })

        trade_df["buy_amount"] = trade_df["buys"].mul(trade_df["buy_price"])
        trade_df["sell_amount"] = trade_df["sells"].mul(trade_df["sell_price"])
        trade_df["Pnl"] = (trade_df["buy_amount"].mul(-1) + trade_df["sell_amount"]).cumsum()
        trade_df["end_positions"] = trade_df["positions"].shift(-1).fillna(0)
        # ----- END ------
        end_time = time.time()
        print("Finished trading : elasped time : " + str(end_time - start_time))
        return trade_df


class MarketMakerSimulator:
    def __init__(self, trade_simulator : TradeSimulator, ask_reference : str, bid_reference : str
                 , threshold : float, tick_size : float, position_max_duration : int, limit_order_standing : int
                 , take_profit : float, stop_loss : float, size : float):
        """
        Strategy : at the end of every bar, send 2 limit orders, a sell and a buy. in the event of inventory excess, lower bid price
        to discourage more buys. All orders have fixed max duration and take profit and stop loss.
        :param trade_simulator: trade simulator object
        :param ask_reference: name of column of the bar data frame used as a reference to determine sell limit order price
        :param bid_reference: name of column of the bar data frame used as a reference to determine buy limit order price
        :param threshold: threshold, the amount of imbalance per raising of the limit price by 1
        :param tick_size: the minimum price increment of the market
        :param position_max_duration: the maximum hold duration of a position before closing it with a market order
        :param limit_order_standing: the duration of the limit order before canceling it
        :param take_profit: handling of the take profit is done by the trade simulator
        :param stop_loss: handling of stop loss is done by the trade simulator
        :param size: the size of trade
        """
        self.trade_simulator = trade_simulator
        self.ask_reference = ask_reference
        self.bid_reference = bid_reference
        self.threshold = threshold
        self.tick_size = tick_size
        self.position_max_duration = position_max_duration
        self.limit_order_standing = limit_order_standing
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.size = size

        self.trade_simulator.reset()

    def trade(self, bar_df : pd.DataFrame):
        bar_df.index = pd.RangeIndex(len(bar_df))
        for bar_no, row in bar_df.iterrows():
            if bar_no >= len(bar_df) - 1:
                self.close_last_positions(row, bar_no)
                break
            self.trade_simulator.trade(row, bar_no)
            self.trade_simulator.match_positions(row, bar_no)
            self.submit_orders(row, bar_no)


    def submit_orders(self, row : pd.Series, bar_no : int) -> None:
        imbalance = self.trade_simulator.get_aggregate_position()
        ask_ref = max(row[self.ask_reference], row[definitions.OPEN_BID] + self.tick_size)
        bid_ref = min(row[self.bid_reference], row[definitions.OPEN_ASK] - self.tick_size)
        ask_price = ask_ref if imbalance >= 0 else ask_ref + abs(imbalance) * self.tick_size
        bid_price = bid_ref if imbalance <= 0 else bid_ref - abs(imbalance) * self.tick_size
        if ask_price == bid_price:
            return
        self.trade_simulator.submit_limit_order(1, self.size, bid_price, self.take_profit, self.stop_loss,
                                                self.position_max_duration, bar_no, row[definitions.TIMESTAMP], self.limit_order_standing)
        self.trade_simulator.submit_limit_order(-1, self.size, ask_price, self.take_profit, self.stop_loss,
                                                self.position_max_duration, bar_no, row[definitions.TIMESTAMP], self.limit_order_standing)

    def close_last_positions(self, row : pd.Series, bar_no : int) -> None:
        self.trade_simulator.close_last_positions(row, bar_no)

    def get_results(self) -> pd.DataFrame:
        return self.trade_simulator.get_results()

    def reset(self) -> None:
        self.trade_simulator.reset()
