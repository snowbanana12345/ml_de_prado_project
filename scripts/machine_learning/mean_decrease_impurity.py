import definitions
from sklearn.ensemble import RandomForestClassifier
from machine_learning.feature_generators import TaLibGenerator
from machine_learning.feature_selectors import ClassificationMdi
from machine_learning.label_generators import DoubleBarrierLabel
from machine_learning.filter_generators import EveryKthFilter
from data_base.data_base import instance as db
import matplotlib.pyplot as plt
import numpy as np

root_dir = definitions.ROOTDIR
# ----- inputs ----
n_estimators = 1000

look_ahead = 15
upper_barrier = 5
lower_barrier = 5
filter_period = 1

atr_time_period = 15
adx_time_period = 15
apo_fast_period = 4
apo_slow_period = 12
rsi_time_period = 30
roc_time_period = 30
ult_osc_period_1 = 5
ult_osc_period_2 = 10
ult_osc_period_3 = 15
tsf_time_period = 15
p_corr_time_period = 15
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9

date_lst = ["20170125", "20170126", "20170127", "20170131", "20170201", "20170202", "20170203", "20170206", "20170207"
            , "20170208",  "20170209",  "20170210",  "20170213", "20170214", "20170215", "20170216", "20170217", "20170220"]
#date_lst = ["20170125", "20170126"]
bar_size = "100"
period = "morning"
# -------- Initialize objects ------------
helper_rfr = RandomForestClassifier(n_estimators = 1000, max_features = 1)
feature_generator = TaLibGenerator(atr_time_period=atr_time_period, adx_time_period=adx_time_period,
                                   apo_fast_period=apo_fast_period,
                                   rsi_time_period=rsi_time_period, roc_time_period=roc_time_period,
                                   ult_osc_period_1=ult_osc_period_1, ult_osc_period_2=ult_osc_period_2,
                                   tsf_time_period=tsf_time_period, p_corr_time_period=p_corr_time_period,
                                   macd_fast_period=macd_fast_period, macd_slow_period=macd_slow_period,
                                   macd_signal_period=macd_signal_period)
label_generator = DoubleBarrierLabel(look_ahead=look_ahead, upper_barrier=upper_barrier, lower_barrier=lower_barrier)
filter_generator = EveryKthFilter(period=filter_period)
feature_selector = ClassificationMdi(feature_generator = feature_generator, label_generator = label_generator,
                                     filter_generator = filter_generator, random_forest = helper_rfr)

result_dfs = []
# --------- run mdi on individual dates ----------
for date in date_lst:
    bar_df = db.get_vol_bar(date, period, bar_size)
    feature_selector.reset()
    feature_selector.append_train_data_set(bar_df)
    feature_selector.pre_process()
    feature_selector.fit()
    result_dfs.append(feature_selector.get_mdi())

# ------- plot feature importance stability ------
for i in range(len(feature_selector.label_list)):
    plt.figure()
    label = feature_selector.label_list[i]
    plt.title(label)
    feat_impt_lst = [result_lst[i] for result_lst in result_dfs]
    for feature in feature_selector.feature_list:
        feat_impt = []
        for result_df in feat_impt_lst:
            feat_impt.append(result_df.loc[feature, "mean"])
        plt.plot(feat_impt, label = feature)
    plt.xticks(range(len(date_lst)), date_lst)
    plt.legend()
plt.show()

# ------- print out statistics -------
for i in range(len(feature_selector.label_list)):
    label = feature_selector.label_list[i]
    print("---- Label : " + label + " ----")
    feat_impt_lst = [result_lst[i] for result_lst in result_dfs]
    for feature in feature_selector.feature_list:
        feat_impt = []
        for result_df in feat_impt_lst:
            feat_impt.append(result_df.loc[feature, "mean"])
        print(feature + "\t" + str(np.mean(feat_impt)) + "\t" + str(np.std(feat_impt)))
# ------ END ------
