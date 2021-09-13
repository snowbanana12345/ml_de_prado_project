import definitions
from sklearn.ensemble import RandomForestClassifier
from machine_learning.feature_generators import TaLibGenerator
from machine_learning.feature_selectors import ClassificationMdaMdi
from machine_learning.label_generators import DoubleBarrierLabel
from machine_learning.filter_generators import EveryKthFilter
from data_base.data_base import instance as db


root_dir = definitions.ROOTDIR
# ----- inputs ----
n_estimators = 1000
train_size = 3
test_size = 1

u_n_estimators = 250
u_max_depth = 10
u_tree_min_sample_leaf = 5
u_tree_min_sample_split = 5
u_max_features = 6
u_random_state = 4
u_class_weight = "balanced"

look_ahead = 15
upper_barrier = 15
lower_barrier = 15
filter_period = 3

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

#date_lst = ["20170125", "20170126", "20170127", "20170131", "20170201", "20170202", "20170203", "20170206", "20170207"
#            , "20170208",  "20170209",  "20170210",  "20170213", "20170214", "20170215", "20170216", "20170217", "20170220"]
date_lst = ["20170125", "20170126", "20170127", "20170131"]
bar_size = "100"
period = "morning"
sampling = "volume"
# -------- Initialize objects ------------
#helper_rfr = RandomForestClassifier(n_estimators = 1000, max_features = 1)
helper_rfr = RandomForestClassifier(n_estimators=u_n_estimators, max_depth=u_max_depth,
                                     min_samples_leaf=u_tree_min_sample_leaf
                                     , random_state=u_random_state, min_samples_split=u_tree_min_sample_split,
                                     max_features=u_max_features, class_weight=u_class_weight)
feature_generator = TaLibGenerator(atr_time_period=atr_time_period, adx_time_period=adx_time_period,
                                   apo_fast_period=apo_fast_period,
                                   rsi_time_period=rsi_time_period, roc_time_period=roc_time_period,
                                   ult_osc_period_1=ult_osc_period_1, ult_osc_period_2=ult_osc_period_2,
                                   tsf_time_period=tsf_time_period, p_corr_time_period=p_corr_time_period,
                                   macd_fast_period=macd_fast_period, macd_slow_period=macd_slow_period,
                                   macd_signal_period=macd_signal_period)
label_generator = DoubleBarrierLabel(look_ahead=look_ahead, upper_barrier=upper_barrier, lower_barrier=lower_barrier)
filter_generator = EveryKthFilter(period=filter_period)
feature_selector = ClassificationMdaMdi(feature_generator = feature_generator, label_generator = label_generator,
                                     filter_generator = filter_generator, base_model = helper_rfr)

# -------- run mda with cross validation ---------
mda_results = []
mdi_results = []
for i in range(len(date_lst) - train_size - test_size + 1):
    feature_selector.reset()
    for j in range(train_size):
        feature_selector.append_train_data_set(db.get_bar_wlimit(date_lst[i + j], period, bar_size, sampling))
    for j in range(train_size, train_size + test_size, 1):
        feature_selector.append_test_data_set(db.get_bar_wlimit(date_lst[i + j], period, bar_size, sampling))
    feature_selector.pre_process()
    feature_selector.fit()
    mda_results.append(feature_selector.get_mda())
    mdi_results.append(feature_selector.get_mdi())

# ------- print mda results ------
print("# ------ Mean decrease accuracy results ------ #")
for date, result in zip(date_lst, mda_results):
    for label in result:
        print("----- Results for label : " + label + " -------")
        label_result = result[label]
        for feature in label_result:
            print(feature, label_result[feature])

# ------- print mdi results ------
print("# ------ Mean decrease impurity results ------ #")
for date, result in zip(date_lst, mdi_results):
    print(" ----- results for date : " + date + " ----- ")
    for label in result:
        print("----- Results for label : " + label + " -------")
        label_result = result[label]
        print(label_result)






