import pandas as pd
import time

class FilterGenerator:
    def __init__(self):
        pass

    def create_filter(self, bar_df : pd.DataFrame) -> pd.Series:
        raise Exception("called abstract method : create_filters of abstract class : FilterGenerator")

    def get_filter_name(self) -> str:
        raise Exception("called abstract method : get_filter_name of abstract class : FilterGenerator")


class EveryKthFilter(FilterGenerator):
    def __init__(self, period : int):
        super().__init__()
        self.period = period
        self.filter_name = "every_" + str(self.period) + "th_filter"

    def create_filter(self, bar_df : pd.DataFrame) -> pd.Series:
        start_time = time.time()
        # ----- START ------
        filter_series = pd.Series([True if i % self.period == 0 else False for i in range(len(bar_df))], index = bar_df.index, name = self.filter_name)
        filter_series.name = self.filter_name
        # ---- END -----
        end_time = time.time()
        print("Created filtered : elapsed time : " + str(end_time - start_time))
        return filter_series

    def get_filter_name(self) -> str:
        return self.filter_name

class CumSumFilter(FilterGenerator):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def create_filter(self, bar_df : pd.DataFrame):
        pass


