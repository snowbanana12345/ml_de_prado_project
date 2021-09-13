import pandas as pd
import numpy as np
import time

def nano_time_to_date_time(ts):
    return (pd.Timestamp(ts) + pd.to_timedelta(8, unit = "h"))

def pre_process(unprocessed_tick_df):
    """ remove outliers and set date time as index
    NOTE : it has to be date time in order to do time sampling as pandas resample cant compare times """
    start_time = time.time()
    tick_df = unprocessed_tick_df.loc[unprocessed_tick_df.ask1p > 17000].copy()
    tick_df["time of day"] = tick_df["timestampNano"].apply(nano_time_to_date_time)
    tick_df.set_index("time of day", inplace = True)
    end_time = time.time()
    print("Removed outliers < 17000 and set date time as index : elasped time : " + str(end_time - start_time))
    return tick_df

def morning_after_noon_split(tick_df : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """ search for the large time gap of roughly 1 hour and split the single day data frame
    NOTE : make sure tick_df is ranged indexed, DO NOT touch the index unless just before time sampling """
    start_time = time.time()
    nanotime_diff = tick_df["timestampNano"].diff().shift(-1)
    pointer = nanotime_diff.idxmax() + 1
    morning_df = tick_df[0: pointer]
    afternoon_df = tick_df[pointer:]
    # ------ END ------
    end_time = time.time()
    print("Data split into morning and afternoon : elapsed time : " + str(end_time - start_time) + " s")
    return morning_df, afternoon_df

def morning_afternoon_midnight_split(tick_df : pd.DataFrame) ->(pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ divide daily tick data into three regimes at two points, 2:30 pm, 2:00 am the next day assuming a start time of 7:30 am """
    start_time = time.time()
    # ------- START ------
    start_time_nano = tick_df.loc[0, "timestampNano"] # ~7:30 am
    split_point_1 = start_time_nano + 7 * 3600 * 1E9 # 2:30 pm
    split_point_2 = start_time_nano + 18 * 3600 * 1E9 + 1800 * 1E9 # 2:00 am
    region_1 = tick_df["timestampNano"] < split_point_1
    region_1_p = np.logical_not(region_1)
    region_2 = tick_df["timestampNano"] < split_point_2
    region_2_p = np.logical_not(region_2)
    morning_df = tick_df[region_1]
    afternoon_df = tick_df[np.logical_and(region_2, region_1_p)]
    midnight_df = tick_df[region_2_p]
    # ------- END -------
    end_time = time.time()
    print("Data split into morning, afternoon and midnight : elapsed time : " + str(end_time - start_time) + " s")
    return morning_df, afternoon_df, midnight_df

def time_sampling(tick_df, rule ="60S", last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano", save = None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame(
                {"open" : [], "close" : [], "high" : [], "low" : [], "total_volume" : [], "VVAP" : [], "timestamp" : [], "date_time" : []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """ resample into time bars, it is assumed that the input data frame already has a valid date time as its index"""
    price_col = tick_df[last_price_col]
    qty_col = tick_df[last_qty_col]
    price_col = price_col[price_col > 0]
    qty_col = qty_col[qty_col > 0]
    timestamps = tick_df[time_stamp_col]
    open_price = price_col.resample(rule).first()
    close_price = price_col.resample(rule).last()
    high_price = price_col.resample(rule).max()
    low_price = price_col.resample(rule).min()
    total_vol = qty_col.resample(rule).sum()
    timestamps = timestamps.resample(rule).first()
    date_time = timestamps.apply(nano_time_to_date_time)
    vol_price_product = (price_col * qty_col).resample(rule).sum()
    volume_weighted_average_price = vol_price_product.div(total_vol)
    new_df = pd.concat([open_price, close_price, high_price, low_price, total_vol, volume_weighted_average_price, timestamps, date_time], axis = 1)
    new_df.columns = ["open", "close", "high", "low", "total_volume", "VVAP", "timestamp", "date_time"]
    new_df.fillna(0, inplace = True)
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x : x == 0).interpolate().ffill().bfill()
    if save :
        new_df.to_csv(save, index = False)
    end_time = time.time()
    print("Time sampling completed : elasped time : " + str(end_time - start_time))
    return new_df

def volume_sampling(tick_df, bar_size = 50, last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano", save = None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame(
                {"timestamp": [], "open": [], "close": [], "high": [], "low": [], "VVAP": [], "volume_time" : []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """
    combines tick data into volume bars of size specified by bar_size
    NOTE : the last bit of data that does not form a bar is dropped
    """
    df = tick_df[tick_df[last_price_col] > 0]
    new_df = pd.DataFrame({"timestamp":[], "open":[], "close":[], "high":[], "low":[], "VVAP":[]})
    new_bar = True
    volume_counter = 0
    time_stamp = 0
    curr_prices = []
    curr_volumes = []
    for i,row in df.iterrows():
        while volume_counter >= bar_size:
            volume_counter = volume_counter - bar_size
            open_price = curr_prices[0]
            close_price = curr_prices[-1]
            high_price = max(curr_prices)
            low_price = min(curr_prices)
            VVAP = sum([vol * price for vol, price in zip(curr_volumes, curr_prices)]) / bar_size
            new_bar = {"timestamp": time_stamp, "open" : open_price, "close": close_price
                , "high" : high_price, "low" : low_price, "VVAP" : VVAP}
            new_df = new_df.append(new_bar, ignore_index = True)
            if volume_counter == 0: # if there is no left over volume, we have hit exactly bar_size amount.
                curr_prices = []
                curr_volumes = []
            else : # there is still some volume left over
                curr_prices = [close_price, ] # the close price of previous bar becomes the new open price
                curr_volumes = [min(volume_counter, bar_size)] # if spill over volume exceeds bar_size amount
            new_bar = True
        if new_bar:
            time_stamp = row[time_stamp_col]
            new_bar = False
        volume_counter = volume_counter + row[last_qty_col]
        curr_prices.append(row[last_price_col])
        if volume_counter >= bar_size:
            curr_volumes.append(bar_size - volume_counter + row[last_qty_col])
        else :
            curr_volumes.append(row[last_qty_col])
    new_df["volume_time"] = list(range(0, len(new_df) * bar_size, bar_size))
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x : x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index = False)
    end_time = time.time()
    print("Volume sampling completed : elasped time : " + str(end_time - start_time))
    return new_df

def tick_sampling(tick_df, bar_size = 20, last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano", save = None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "volume": [], "VVAP" : [], "tick_time" : []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """ Combines tick data into bunches of bar_size number of ticks, 
        only ticks with non zero trade volume is counted 
    NOTE : left over ticks are combined into the last bar """
    price_col = tick_df[last_price_col]
    qty_col = tick_df[last_qty_col]
    time_col = tick_df[time_stamp_col]
    # --------- filter out 0 rows --------
    time_col = time_col[qty_col > 0]
    price_col = price_col[price_col > 0]
    qty_col = qty_col[qty_col > 0]
    # --------- reindex to a range index --------
    price_col.index = pd.RangeIndex(0, len(price_col), 1)
    qty_col.index = pd.RangeIndex(0, len(qty_col), 1)
    time_col.index = pd.RangeIndex(0, len(time_col), 1)
    # -------- find bar features -------
    open_price = price_col.groupby(price_col.index // bar_size).first()
    close_price = price_col.groupby(price_col.index // bar_size).last()
    high_price = price_col.groupby(price_col.index // bar_size).max()
    low_price = price_col.groupby(price_col.index // bar_size).min()
    vol_price_product = price_col.mul(qty_col).groupby(price_col.index // bar_size).sum()
    total_vol = qty_col.groupby(qty_col.index // bar_size).sum()
    vvap_col = vol_price_product.div(total_vol)
    # ------- process time column -------
    time_col = time_col.groupby(time_col.index // bar_size).first()
    # ------- create and save the new data frame -------
    new_df = pd.DataFrame({
        "open" : open_price,
        "close" : close_price,
        "high" : high_price,
        "low" : low_price,
        "VVAP" : vvap_col,
        "volume" : total_vol,
        "timestamp" : time_col
    })
    new_df["tick_time"] = [*range(0, len(new_df) * bar_size, bar_size)]
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x : x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index = False)
    end_time = time.time()
    print("Tick sampling completed : elasped time : " + str(end_time - start_time))
    return new_df

def dollar_sampling(tick_df, bar_size = 1000000, last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano", save = None):
    start_time = time.time()
    """
    combines tick data into dollar bars of size specified by bar_size
    NOTE : the last bit of data that does not form a bar is dropped
    """
    df = tick_df[tick_df[last_price_col] > 0]
    new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "volume": [], "dollar_time" : []})
    if len(tick_df) == 0:
        if save :
            new_df.to_csv(save)
        return tick_df
    new_bar = True
    dollar_counter = 0
    prev_dollar_counter = 0
    time_stamp = 0
    curr_prices = []
    curr_volumes = []
    for i,row in df.iterrows():
        while dollar_counter >= bar_size: # this loop will be entered multiple times when a large tick comes that fills multiple bars
            dollar_counter = dollar_counter - bar_size
            open_price = curr_prices[0]
            close_price = curr_prices[-1]
            high_price = max(curr_prices)
            low_price = min(curr_prices)
            last_tick_total_volume = curr_volumes[-1]
            last_tick_fill = bar_size - prev_dollar_counter # the amount of the dollar bar filled by the last tick
            prev_tick_volumes = sum(curr_volumes[:-1]) # the volume of all ticks except the last volume
            last_tick_volume = last_tick_fill / close_price # the volume filled by the last tick
            bar_volume = last_tick_volume + prev_tick_volumes # the total volume in this bar
            new_bar = {"timestamp": time_stamp, "open": open_price, "close": close_price
                , "high": high_price, "low": low_price, "volume" : bar_volume}
            new_df = new_df.append(new_bar, ignore_index = True)
            if dollar_counter >= bar_size : # the tick has so much dollar that it can fill multiple bars
                curr_prices = [close_price,]
                curr_volumes = [last_tick_total_volume - last_tick_volume]
                prev_dollar_counter = 0
            elif dollar_counter == 0 : # the bar filled exactly
                curr_prices = []
                curr_volumes = []
                prev_dollar_counter = 0
            else : # the is a left over amount < bar_size which is to be carried forward to the next bar
                curr_prices = [close_price,]
                curr_volumes = [last_tick_total_volume - last_tick_volume, ]
                prev_dollar_counter = dollar_counter
            new_bar = True
        if new_bar :
            time_stamp = row[time_stamp_col]
            new_bar = False
        dollar_counter = dollar_counter + row[last_qty_col] * row[last_price_col]
        curr_prices.append(row[last_price_col])
        curr_volumes.append(row[last_qty_col])
        if dollar_counter < bar_size:
            prev_dollar_counter = dollar_counter
    new_df["dollar_time"] = list(range(0, len(new_df) * bar_size, bar_size))
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x : x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index = False)
    end_time = time.time()
    print("Dollar sampling completed : elasped time : " + str(end_time - start_time))
    return new_df

def volume_sampling_limit_book(tick_df, bar_size, last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano",
                               best_ask_col = "best_ask",  best_bid_col = "best_bid", save = None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "VVAP": [],
                                   "open_bid": [], "high_bid": [], "low_bid": [], "close_bid": [], "average_bid": [],
                                   "open_ask": [], "close_ask": [], "high_ask": [], "low_ask": [], "average_ask": [],
                                   "average_bid_ask_spread": []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """
    combines tick data into volume bars of size specified by bar_size
    NOTE : the last bit of data that does not form a bar is dropped
    NOTE : 
    """
    df = tick_df[tick_df[last_price_col] > 0]
    new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "VVAP": [],
                           "open_bid" : [], "high_bid" : [], "low_bid" : [], "close_bid" : [], "average_bid" : [],
                           "open_ask" : [], "close_ask" : [], "high_ask" : [], "low_ask" : [], "average_ask" : [],
                           "average_bid_ask_spread" : [], "std_bid_ask_spread" : []})
    new_bar = True
    volume_counter = 0
    time_stamp = 0
    curr_prices = []
    curr_volumes = []
    best_asks = []
    best_bids = []
    bid_ask_spreads = []
    for i, row in df.iterrows():
        while volume_counter >= bar_size:
            # ----- trade statistics ------
            volume_counter = volume_counter - bar_size
            open_price = curr_prices[0]
            close_price = curr_prices[-1]
            high_price = max(curr_prices)
            low_price = min(curr_prices)
            # ----- bid ask statistics ------
            open_bid = best_bids[0]
            close_bid = best_bids[-1]
            high_bid = max(best_bids)
            low_bid = min(best_bids)
            average_bid = np.mean(best_bids)

            open_ask = best_asks[0]
            close_ask = best_asks[-1]
            high_ask = max(best_asks)
            low_ask = min(best_asks)
            average_ask = np.mean(best_asks)

            average_bid_ask_spread = np.mean(bid_ask_spreads)
            std_bid_ask_spread = np.std(bid_ask_spreads)

            VVAP = sum([vol * price for vol, price in zip(curr_volumes, curr_prices)]) / bar_size
            new_bar = {"timestamp": time_stamp, "open": open_price, "close": close_price,"high": high_price, "low": low_price, "VVAP": VVAP,
                       "open_bid" : open_bid, "close_bid" : close_bid, "high_bid" : high_bid, "low_bid" : low_bid, "average_bid" : average_bid,
                       "open_ask": open_ask, "close_ask": close_ask, "high_ask": high_ask, "low_ask": low_ask, "average_ask": average_ask,
                       "average_bid_ask_spread" : average_bid_ask_spread, "std_bid_ask_spread" : std_bid_ask_spread
                       }
            new_df = new_df.append(new_bar, ignore_index=True)
            if volume_counter == 0:  # if there is no left over volume, we have hit exactly bar_size amount.
                curr_prices = []
                curr_volumes = []
                best_asks = []
                best_bids = []
                bid_ask_spreads = []
            else:  # there is still some volume left over
                curr_prices = [close_price, ]  # the close price of previous bar becomes the new open price
                curr_volumes = [min(volume_counter, bar_size)]  # if spill over volume exceeds bar_size amount
                best_asks = [close_ask, ]
                best_bids = [close_bid, ]
                bid_ask_spreads = bid_ask_spreads[-1:]
            new_bar = True
        if new_bar:
            time_stamp = row[time_stamp_col]
            new_bar = False
        volume_counter = volume_counter + row[last_qty_col]
        curr_prices.append(row[last_price_col])
        if volume_counter >= bar_size:
            curr_volumes.append(bar_size - volume_counter + row[last_qty_col])
        else:
            curr_volumes.append(row[last_qty_col])
        best_asks.append(row[best_ask_col])
        best_bids.append(row[best_bid_col])
        bid_ask_spreads.append(row[best_ask_col] - row[best_bid_col])

    new_df["volume_time"] = list(range(0, len(new_df) * bar_size, bar_size))
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x: x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index=False)
    end_time = time.time()
    print("Volume sampling completed : elasped time : " + str(end_time - start_time))
    return new_df


def tick_sampling_limit_book(tick_df, bar_size = 20, last_price_col ="lastPrice", last_qty_col ="lastQty", time_stamp_col ="timestampNano",
                             best_ask_col = "best_ask",  best_bid_col = "best_bid", save = None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "volume": [], "VVAP" : [], "tick_time" : [],
                                   "open_bid": [], "high_bid": [], "low_bid": [], "close_bid": [], "average_bid": [],
                                   "open_ask": [], "close_ask": [], "high_ask": [], "low_ask": [], "average_ask": [],
                                   "average_bid_ask_spread": [], "std_bid_ask_spread": []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """ Combines tick data into bunches of bar_size number of ticks, 
        only ticks with non zero trade volume is counted 
    NOTE : left over ticks are combined into the last bar """
    price_col = tick_df[last_price_col]
    qty_col = tick_df[last_qty_col]
    time_col = tick_df[time_stamp_col]
    ask_col = tick_df[best_ask_col]
    bid_col = tick_df[best_bid_col]
    # --------- filter out 0 rows --------
    time_col = time_col[qty_col > 0]
    price_col = price_col[price_col > 0]
    ask_col = ask_col[qty_col > 0]
    bid_col = bid_col[qty_col > 0]
    qty_col = qty_col[qty_col > 0]
    # --------- reindex to a range index --------
    price_col.index = pd.RangeIndex(0, len(price_col), 1)
    qty_col.index = pd.RangeIndex(0, len(qty_col), 1)
    time_col.index = pd.RangeIndex(0, len(time_col), 1)
    ask_col.index = pd.RangeIndex(0, len(qty_col), 1)
    bid_col.index = pd.RangeIndex(0, len(qty_col), 1)
    # -------- find bar features -------
    open_price = price_col.groupby(price_col.index // bar_size).first()
    close_price = price_col.groupby(price_col.index // bar_size).last()
    high_price = price_col.groupby(price_col.index // bar_size).max()
    low_price = price_col.groupby(price_col.index // bar_size).min()
    vol_price_product = price_col.mul(qty_col).groupby(price_col.index // bar_size).sum()
    total_vol = qty_col.groupby(qty_col.index // bar_size).sum()
    vvap_col = vol_price_product.div(total_vol)
    open_ask = ask_col.groupby(price_col.index // bar_size).first()
    close_ask = ask_col.groupby(price_col.index // bar_size).last()
    high_ask = ask_col.groupby(price_col.index // bar_size).max()
    low_ask = ask_col.groupby(price_col.index // bar_size).min()
    average_ask = ask_col.groupby(price_col.index // bar_size).mean()
    open_bid = bid_col.groupby(price_col.index // bar_size).first()
    close_bid = bid_col.groupby(price_col.index // bar_size).last()
    high_bid = bid_col.groupby(price_col.index // bar_size).max()
    low_bid = bid_col.groupby(price_col.index // bar_size).min()
    average_bid = bid_col.groupby(price_col.index // bar_size).mean()
    spread = ask_col - bid_col
    average_bid_ask_spread = spread.groupby(price_col.index // bar_size).mean()
    std_bid_ask_spread = spread.groupby(price_col.index // bar_size).std()
    # ------- process time column -------
    time_col = time_col.groupby(time_col.index // bar_size).first()
    # ------- create and save the new data frame -------
    new_df = pd.DataFrame({
        "open" : open_price,
        "close" : close_price,
        "high" : high_price,
        "low" : low_price,
        "VVAP" : vvap_col,
        "volume" : total_vol,
        "timestamp" : time_col,
        "open_bid": open_bid, "high_bid": high_bid, "low_bid": low_bid, "close_bid": close_bid, "average_bid": average_bid,
        "open_ask": open_ask, "close_ask": close_ask, "high_ask": high_ask, "low_ask": low_ask, "average_ask": average_ask,
        "average_bid_ask_spread": average_bid_ask_spread, "std_bid_ask_spread": std_bid_ask_spread})
    new_df["tick_time"] = [*range(0, len(new_df) * bar_size, bar_size)]
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x : x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index = False)
    end_time = time.time()
    print("Tick sampling completed : elasped time : " + str(end_time - start_time))
    return new_df