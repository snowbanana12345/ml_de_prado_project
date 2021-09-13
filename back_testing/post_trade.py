import pandas as pd
import numpy as np

def get_limit_strategy_statistics(trade_df : pd.DataFrame) -> dict:
    pnl_series = trade_df.loc[trade_df["end_positions"] == 0, 'Pnl']
    total_trade_volume = (sum(trade_df["buys"]) + sum(trade_df["sells"])) / 2
    draw_down = min(pnl_series)
    profit = pnl_series.iloc[-1]
    result = {}
    result["Total_trade_volume"] = total_trade_volume
    result["Max_draw_down"] = draw_down
    result["Profit"] = profit
    return result

def process_trade_df(bar_df : pd.DataFrame, trade_df : pd.DataFrame) -> pd.DataFrame:
    trade_df["profit"] = (trade_df["exit_price"] - trade_df["entry_price"]) * trade_df["direction"] * trade_df["size"]
    profit_df = trade_df[["exit_bar_number", "profit"]]
    profit_df = profit_df.groupby(profit_df["exit_bar_number"]).sum()
    bar_df["profit"] = profit_df["profit"]
    bar_df["profit"] = bar_df["profit"].fillna(0)
    bar_df["pnl"] = bar_df["profit"].cumsum()
    return bar_df

def get_trade_result(bar_df : pd.DataFrame, trade_df : pd.DataFrame) -> dict:
    volume_traded = len(trade_df)
    draw_down = min(bar_df["pnl"])
    profit = bar_df.loc[len(bar_df) - 1, "pnl"]
    std_dev = bar_df["profit"].std()
    avg_profit = profit / len(bar_df)
    sharpie_ratio = avg_profit / std_dev
    result = {}
    result["profit"] = profit
    result["sharpie_ratio"] = sharpie_ratio
    result["volume_traded"] = volume_traded
    result["draw_down"] = draw_down
    return result

