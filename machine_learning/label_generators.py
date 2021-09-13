import pandas as pd
import time
import numpy as np


class LabelGenerator:
    """ abstract class label generator """
    def __init__(self):
        pass

    def create_labels(self, bar_df : pd.DataFrame) -> pd.DataFrame:
        raise Exception("called abstract method : create_labels of abstract class : LabelGenerator")

    def get_label_name(self) -> str:
        raise Exception("called abstract method : get_label_name of abstract class : LabelGenerator")

    def get_label_list(self) -> [str]:
        raise Exception("called abstract method : get_label_list of abstract class : LabelGenerator")


class AbsolutePriceChange(LabelGenerator):
    def __init__(self, look_ahead: int, change_threshold: int, price_col_name : str = "close"):
        super().__init__()
        self.look_ahead = look_ahead
        self.change_threshold = change_threshold
        self.label_name = "abs_change"
        self.price_col_name = price_col_name

    def create_labels(self, bar_df : pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        # ----- START ------
        """ creates a column in the df called delta_label """
        new_df = pd.DataFrame(index = bar_df.index)
        price_series = bar_df[self.price_col_name]
        delta = price_series.shift(- self.look_ahead) - price_series

        def delta_to_label(delt):
            if pd.isna(delt):
                return delt
            if delt > self.change_threshold:
                return 1
            elif delt < - self.change_threshold:
                return -1
            else:
                return 0

        new_df[self.label_name] = delta.apply(delta_to_label)
        # ---- END -----
        end_time = time.time()
        print("Created labels : elapsed time : " + str(end_time - start_time))
        return new_df

    def get_label_name(self) -> str:
        return self.label_name


class DoubleBarrierLabel(LabelGenerator):
    """
    Double Barrier scheme :
    if the price hits the upper barrier above the current price within the look ahead period, label 1 else 0
    if the price hits the lower barrier above the current price within the look ahead period, label 1 else 0

    """
    def __init__(self, look_ahead: int, upper_barrier: float, lower_barrier: float,
                 upper_price_col_name = "high", lower_price_col_name = "low"):
        super().__init__()
        self.look_ahead = look_ahead
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.upper_price_col_name = upper_price_col_name
        self.lower_price_col_name = lower_price_col_name
        self.upper_barrier_name = "upper_barrier"
        self.lower_barrier_name = "lower_barrier"

    def create_labels(self, bar_df : pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        # ----- START ------
        """ create two labels, upper_barrier and lower_barrier, both are bool arrays
        the label is true if the the vvap of a bar reaches upper_barrier amount above the vvap of the current bar else false
        same for the lower barrier """

        def hit_upper(x: np.array) -> int:
            x = x.to_numpy()
            return 1 if np.max(x[:-1] - x[-1]) >= self.upper_barrier else 0

        def hit_lower(x: np.array) -> int:
            x = x.to_numpy()
            return 1 if np.min(x[:-1] - x[-1]) <= -self.lower_barrier else 0

        new_df = pd.DataFrame(index = bar_df.index)
        upper_price_series = bar_df[self.upper_price_col_name]
        lower_price_series = bar_df[self.lower_price_col_name]
        upper_barrier_arr = upper_price_series[::-1].rolling(self.look_ahead + 1).apply(hit_upper)[::-1]
        lower_barrier_arr = lower_price_series[::-1].rolling(self.look_ahead + 1).apply(hit_lower)[::-1]
        new_df[self.upper_barrier_name] = upper_barrier_arr
        new_df[self.lower_barrier_name] = lower_barrier_arr
        # ---- END -----
        end_time = time.time()
        print("Created labels : elapsed time : " + str(end_time - start_time))
        return new_df

    def get_label_list(self) -> [str]:
        return [self.upper_barrier_name, self.lower_barrier_name]

