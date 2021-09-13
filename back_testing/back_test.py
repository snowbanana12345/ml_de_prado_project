import pandas as pd
import definitions


class Position:
    def __init__(self, direction: int, size: float, entry_price: float, take_profit: float, stop_loss: float
                 , max_duration: int, entry_time: int, entry_time_stamp: int):
        """
        :param direction: 1 for buy, -1 for sell
        :param size:
        :param entry_price:
        :param take_profit:
        :param stop_loss:
        :param max_duration: the max number of bars to hold the position
        :param entry_time: the bar number entry
        :param entry_time_stamp: time stamp of the bar the position was entered
        """
        if direction not in [-1, 1]:
            raise ValueError("direction can only be -1 or 1")
        if size <= 0:
            raise ValueError("Position size cannot be 0!")
        self.direction = direction
        self.size = size
        self.entry_price = entry_price
        self.take_profit = self.entry_price + self.direction * take_profit
        self.stop_loss = self.entry_price - self.direction * stop_loss
        self.max_duration = max_duration
        self.entry_time = entry_time
        self.entry_time_stamp = entry_time_stamp
        self.expiry_time = self.entry_time + self.max_duration
        self.borrowed = self.direction * self.size * self.entry_price

    def __str__(self):
        return "Pos:" + str(self.direction) + ":" + str(self.size) + ":" + str(self.entry_price) + ":" + str(
            self.take_profit) \
               + ":" + str(self.stop_loss) + ":" + str(self.expiry_time)


class LimitOrder:
    def __init__(self, direction: int, size: float, price: float, take_profit: float, stop_loss: float,
                 max_duration: int, submit_time: int, submit_time_stamp: int, standing_duration: int):
        if direction not in [-1, 1]:
            raise ValueError("direction can only be -1 or 1")
        self.direction = direction
        self.size = size
        self.price = price
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_duration = max_duration
        self.submit_time = submit_time
        self.submit_time_stamp = submit_time_stamp
        self.standing_duration = standing_duration
        self.expiry_time = self.submit_time + self.standing_duration

    def fill(self, entry_time, entry_time_stamp) -> Position:
        return Position(self.direction, self.size, self.price, self.take_profit, self.stop_loss, self.max_duration,
                        entry_time, entry_time_stamp)

    def __str__(self):
        return "Limit:" + str(self.direction) + ":" + str(self.size) + ":" + str(self.price) + ":" + str(
            self.take_profit) \
               + ":" + str(self.stop_loss) + ":" + str(self.expiry_time)


class MarketOrder:
    def __init__(self, direction: int, size: float, take_profit: float, stop_loss: float,
                 max_duration: int, submit_time: int, submit_time_stamp: int):
        if direction not in [-1, 1]:
            raise ValueError("direction can only be -1 or 1")
        self.direction = direction
        self.size = size
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_duration = max_duration
        self.submit_time = submit_time
        self.submit_time_stamp = submit_time_stamp

    def fill(self, entry_time, entry_time_stamp, bid, ask) -> Position:
        price = ask if self.direction == 1 else bid
        return Position(self.direction, self.size, price, self.take_profit, self.stop_loss, self.max_duration,
                        entry_time, entry_time_stamp)

    def __str__(self):
        return "Mo:" + str(self.direction) + ":" + str(self.size) + ":" + ":" + str(self.take_profit) \
               + ":" + str(self.stop_loss)


class TradeSimulator:
    def __init__(self):
        self.trade_rows = []
        self.market_order_lst = []
        self.limit_order_lst = []
        self.position_lst = []
        self.ask_trigger = "high"
        self.bid_trigger = "low"
        self.tp_buy_trigger = "high"
        self.tp_sell_trigger = "low"
        self.stop_buy_trigger = "low"
        self.stop_sell_trigger = "high"

    def trade(self, row: pd.Series, bar_no: int) -> None:
        self.fill_market_orders(row, bar_no)
        self.check_positions(row, bar_no)
        self.fill_limit_orders(row, bar_no)
        self.close_expired(row, bar_no)

    def fill_market_orders(self, row: pd.Series, bar_no: int) -> None:
        """ fill market orders """
        for market_order in self.market_order_lst:
            self.position_lst.append(market_order.fill(bar_no, row[definitions.TIMESTAMP], row[definitions.OPEN_BID]
                                                       , row[definitions.OPEN_ASK]))

    def check_positions(self, row: pd.Series, bar_no: int) -> None:
        removal = []
        for j, position in enumerate(self.position_lst):
            if position.direction == 1 and position.take_profit <= row[self.tp_buy_trigger]:
                self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                        "direction": position.direction,
                                        "size": position.size, "exit_bar_number": bar_no,
                                        "exit_price": position.take_profit})
                removal.append(j)
            if position.direction == 1 and position.stop_loss >= row[self.stop_buy_trigger]:
                self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                        "direction": position.direction,
                                        "size": position.size, "exit_bar_number": bar_no, "exit_price": row["low_bid"]})
                removal.append(j)
            if position.direction == -1 and position.take_profit >= row[self.tp_sell_trigger]:
                self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                        "direction": position.direction,
                                        "size": position.size, "exit_bar_number": bar_no,
                                        "exit_price": position.take_profit})
                removal.append(j)
            if position.direction == -1 and position.stop_loss <= row[self.stop_sell_trigger]:
                self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                        "direction": position.direction,
                                        "size": position.size, "exit_bar_number": bar_no,
                                        "exit_price": row["high_ask"]})
                removal.append(j)
        self.position_lst = [position for j, position in enumerate(self.position_lst) if j not in removal]

    def fill_limit_orders(self, row: pd.Series, bar_no: int):
        removal = []
        for j, limit_order in enumerate(self.limit_order_lst):
            if limit_order.direction == 1 and row[self.bid_trigger] <= limit_order.price:
                self.position_lst.append(limit_order.fill(bar_no, row["timestamp"]))
                removal.append(j)
            if limit_order.direction == -1 and row[self.ask_trigger] >= limit_order.price:
                self.position_lst.append(limit_order.fill(bar_no, row["timestamp"]))
                removal.append(j)
        self.limit_order_lst = [limit_order for j, limit_order in enumerate(self.limit_order_lst) if j not in removal]

    def close_expired(self, row: pd.Series, bar_no: int):
        removal = []
        for j, position in enumerate(self.position_lst):
            if bar_no == position.expiry_time:
                self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                        "direction": position.direction, "size": position.size,
                                        "exit_bar_number": bar_no,
                                        "exit_price": row["close_ask"] if position.direction == -1 else row[
                                            "close_bid"]})
                removal.append(j)
        self.position_lst = [position for j, position in enumerate(self.position_lst) if j not in removal]
        removal = []
        for j, limit_order in enumerate(self.limit_order_lst):
            if bar_no == limit_order.expiry_time:
                removal.append(j)
        self.limit_order_lst = [limit_order for j, limit_order in enumerate(self.limit_order_lst) if j not in removal]

    def submit_limit_order(self, direction: int, size: float, price: float, take_profit: float, stop_loss: float,
                           max_duration: int, submit_time: int, submit_time_stamp: int, standing_duration: int):
        self.limit_order_lst.append(LimitOrder(direction, size, price, take_profit, stop_loss, max_duration, submit_time
                                               , submit_time_stamp, standing_duration))

    def submit_market_order(self, direction: int, size: float, take_profit: float, stop_loss: float,
                            max_duration: int, submit_time: int, submit_time_stamp: int):
        self.market_order_lst.append(MarketOrder(direction, size, take_profit
                                                 , stop_loss, max_duration, submit_time, submit_time_stamp))

    def close_last_positions(self, row: pd.Series, bar_no: int) -> None:
        self.limit_order_lst = []
        self.market_order_lst = []
        for position in self.position_lst:
            self.trade_rows.append({"entry_bar_number": position.entry_time, "entry_price": position.entry_price,
                                    "direction": position.direction, "size": position.size, "exit_bar_number": bar_no,
                                    "exit_price": row["close_ask"] if position.direction == -1 else row["close_bid"]})
        self.position_lst = []

    def match_positions(self, row: pd.Series, bar_no: int) -> None:
        """
        Matches sell positions with buy positions
        We close the position that arrives sooner, if both buy and sell arrives at the same time, we record it as sell
        If two positions have different sizes, we simulate this by pretending that the larger position of size1 was entered as
        two positions of sizes size1 - size2 and size2 and position of size2 is closed.
        :param row:
        :param bar_no:
        :return:
        """
        self.position_lst = sorted(self.position_lst, key=lambda pos: pos.entry_time + pos.max_duration)
        buy_positions = [position for position in self.position_lst if position.direction == 1]
        sell_positions = [position for position in self.position_lst if position.direction == -1]
        while buy_positions and sell_positions:
            buy_pos = buy_positions.pop(0)
            sell_pos = sell_positions.pop(0)
            if buy_pos.size == sell_pos.size:
                if buy_pos.entry_time < sell_pos.entry_time:
                    self.trade_rows.append(
                        {"entry_bar_number": buy_pos.entry_time, "entry_price": buy_pos.entry_price,
                         "direction": buy_pos.direction, "size": buy_pos.size,
                         "exit_bar_number": bar_no, "exit_price": sell_pos.entry_price})
                else:
                    self.trade_rows.append(
                        {"entry_bar_number": sell_pos.entry_time, "entry_price": sell_pos.entry_price,
                         "direction": sell_pos.direction, "size": sell_pos.size,
                         "exit_bar_number": bar_no, "exit_price": buy_pos.entry_price})
            elif buy_pos.size < sell_pos.size:
                if buy_pos.entry_time < sell_pos.entry_time:
                    self.trade_rows.append(
                        {"entry_bar_number": buy_pos.entry_time, "entry_price": buy_pos.entry_price,
                         "direction": buy_pos.direction, "size": buy_pos.size,
                         "exit_bar_number": bar_no, "exit_price": sell_pos.entry_price})
                else:
                    self.trade_rows.append(
                        {"entry_bar_number": sell_pos.entry_time, "entry_price": sell_pos.entry_price,
                         "direction": sell_pos.direction, "size": buy_pos.size,
                         "exit_bar_number": bar_no, "exit_price": buy_pos.entry_price})
                sell_positions.insert(0, Position(direction=-1, size=sell_pos.size - buy_pos.size,
                                                  entry_price=sell_pos.entry_price,
                                                  take_profit=sell_pos.take_profit, stop_loss=sell_pos.stop_loss,
                                                  max_duration=sell_pos.max_duration,
                                                  entry_time=sell_pos.entry_time,
                                                  entry_time_stamp=sell_pos.entry_time_stamp))
            else:
                if buy_pos.entry_time < sell_pos.entry_time:
                    self.trade_rows.append(
                        {"entry_bar_number": sell_pos.entry_time, "entry_price": sell_pos.entry_price,
                         "direction": sell_pos.direction, "size": buy_pos.size,
                         "exit_bar_number": bar_no, "exit_price": sell_pos.entry_price})
                else:
                    self.trade_rows.append(
                        {"entry_bar_number": sell_pos.entry_time, "entry_price": sell_pos.entry_price,
                         "direction": sell_pos.direction, "size": buy_pos.size,
                         "exit_bar_number": bar_no, "exit_price": sell_pos.entry_price})
                buy_positions.insert(0, Position(direction=1, size = buy_pos.size - sell_pos.size,
                                                  entry_price=buy_pos.entry_price,
                                                  take_profit=buy_pos.take_profit, stop_loss=buy_pos.stop_loss,
                                                  max_duration=buy_pos.max_duration,
                                                  entry_time=buy_pos.entry_time,
                                                  entry_time_stamp = buy_pos.entry_time_stamp))

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_rows)

    def reset(self):
        self.trade_rows = []
        self.market_order_lst = []
        self.limit_order_lst = []
        self.position_lst = []

    # ------ helper functions ------
    def get_aggregate_position(self) -> float:
        imbalance = 0
        for position in self.position_lst:
            imbalance += position.direction * position.size
        return imbalance


