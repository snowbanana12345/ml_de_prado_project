import pandas as pd
import time
import numpy as np
import talib


class FeatureGenerator:
    """ abstract class feature generator """
    def __init__(self):
        pass

    def create_features(self, bar_df : pd.DataFrame) -> pd.DataFrame:
        raise Exception("called abstract method : create_features of abstract class : FeatureGenerator")

    def get_feature_list(self) -> [str]:
        raise Exception("called abstract method : get_feature_list of abstract class : FeatureGenerator")


class TaLibGenerator(FeatureGenerator):
    def __init__(self, atr_time_period=15, adx_time_period=15, apo_fast_period=4, apo_slow_period=12,
                 rsi_time_period=30, roc_time_period=30, ult_osc_period_1=5, ult_osc_period_2=10, ult_osc_period_3=15,
                 tsf_time_period=15, p_corr_time_period=15, macd_fast_period=12, macd_slow_period=26,
                 macd_signal_period=9):
        super().__init__()
        self.atr_time_period = atr_time_period
        self.adx_time_period = adx_time_period
        self.apo_fast_period = apo_fast_period
        self.apo_slow_period = apo_slow_period
        self.rsi_time_period = rsi_time_period
        self.roc_time_period = roc_time_period
        self.ult_osc_period_1 = ult_osc_period_1
        self.ult_osc_period_2 = ult_osc_period_2
        self.ult_osc_period_3 = ult_osc_period_3
        self.tsf_time_period = tsf_time_period
        self.p_corr_time_period = p_corr_time_period
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.feature_list = ["atr", "adx", "apo", "rsi", "roc", "ult_osc", "log_tsf", "per_corr", "macd", "macd_signal", "macd_hist"]

    def create_features(self, bar_df : pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        # ----- START ------
        """ create features using technical indicators """
        high = bar_df["high"]
        low = bar_df["low"]
        close = bar_df["close"]

        new_bar_df = pd.DataFrame(index = bar_df.index)

        """ atr is the average true range , volatility indicator , gives info as to if a price movement is coming """
        new_bar_df["atr"] = talib.ATR(high, low, close, timeperiod= self.atr_time_period)
        """ average directional index, adx > 25 indicates a strong trend, adx < 20 indicates a weak trend """
        new_bar_df["adx"] = talib.ADX(high, low, close, timeperiod=self.adx_time_period)
        """ difference between two exponential moving averages, APO > 0 indicates up trend, APO < 0 indicates down trend  
        Does not indicate price reversals from a new high or new low """
        new_bar_df["apo"] = talib.APO(close, fastperiod=self.apo_fast_period, slowperiod=self.apo_slow_period, matype=0)
        """ rsi > 0.7 means overbought , rsi < 0.3 means over sold"""
        new_bar_df["rsi"] = talib.RSI(close, timeperiod=self.rsi_time_period)
        """ rate of change """
        new_bar_df["roc"] = talib.ROC(close, timeperiod=self.roc_time_period)
        """ A version of the rsi, uses 3 time frames, smoothens out short term divergences , generates less signals """
        new_bar_df["ult_osc"] = talib.ULTOSC(high, low, close, timeperiod1=self.ult_osc_period_1, timeperiod2=self.ult_osc_period_2,
                                         timeperiod3=self.ult_osc_period_3)
        """ time series forcast, whatever that is """
        new_bar_df["log_tsf"] = log_diff(talib.TSF(close, timeperiod=self.tsf_time_period))
        """ person correlation, presumably this is a correlation between high and low of a bar """
        new_bar_df["per_corr"] = talib.CORREL(high, low, timeperiod=self.p_corr_time_period)
        """ moving average convergence divergence, calculates the difference between two moving averages """
        new_bar_df["macd"], new_bar_df["macd_signal"], new_bar_df["macd_hist"] = talib.MACD(close, fastperiod=self.macd_fast_period,
                                                                                slowperiod=self.macd_slow_period,
                                                                                signalperiod=self.macd_signal_period)
        # ------- END -------
        end_time = time.time()
        print("Created features : elasped time : " + str(end_time - start_time))
        return new_bar_df

    def get_feature_list(self) -> [str]:
        return self.feature_list

def log_diff(feature : pd.Series)->pd.Series:
    """ converts a non stationary series to stationary series by taking log difference """
    return np.log(feature).diff()