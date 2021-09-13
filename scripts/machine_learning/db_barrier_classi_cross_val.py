import definitions
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from machine_learning.double_barrier_models import DoubleBarrierClassificationModel
from machine_learning.feature_generators import TaLibGenerator
from machine_learning.label_generators import DoubleBarrierLabel
from machine_learning.filter_generators import EveryKthFilter
from machine_learning.model_selector import DataSelector

root_dir = definitions.ROOTDIR
# ----- inputs ----
look_ahead = 15
upper_barrier = 5
lower_barrier = 5
filter_period = 1

u_n_estimators = 250
u_max_depth = 10
u_tree_min_sample_leaf = 5
u_tree_min_sample_split = 5
u_max_features = 6
u_random_state = 4
u_class_weight = "balanced"

l_n_estimators = 250
l_max_depth = 10
l_tree_min_sample_leaf = 10
l_tree_min_sample_split = 5
l_max_features = 8
l_random_state = 4
l_class_weight = "balanced"

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

# -------- Initialize objects ------------
upper_model = RandomForestClassifier(n_estimators=u_n_estimators, max_depth=u_max_depth,
                                     min_samples_leaf=u_tree_min_sample_leaf
                                     , random_state=u_random_state, min_samples_split=u_tree_min_sample_split,
                                     max_features=u_max_features, class_weight=u_class_weight)
lower_model = RandomForestClassifier(n_estimators=l_n_estimators, max_depth=l_max_depth,
                                     min_samples_leaf=l_tree_min_sample_leaf
                                     , random_state=l_random_state, min_samples_split=l_tree_min_sample_split,
                                     max_features=l_max_features, class_weight=l_class_weight)
feature_generator = TaLibGenerator(atr_time_period=atr_time_period, adx_time_period=adx_time_period,
                                   apo_fast_period=apo_fast_period,
                                   rsi_time_period=rsi_time_period, roc_time_period=roc_time_period,
                                   ult_osc_period_1=ult_osc_period_1, ult_osc_period_2=ult_osc_period_2,
                                   tsf_time_period=tsf_time_period, p_corr_time_period=p_corr_time_period,
                                   macd_fast_period=macd_fast_period, macd_slow_period=macd_slow_period,
                                   macd_signal_period=macd_signal_period)
label_generator = DoubleBarrierLabel(look_ahead=look_ahead, upper_barrier=upper_barrier, lower_barrier=lower_barrier)
filter_generator = EveryKthFilter(period=filter_period)
model = DoubleBarrierClassificationModel(upper_estimator = upper_model,
                                         lower_estimator = lower_model,
                                         feature_generator=feature_generator,
                                         label_generator=label_generator,
                                         filter_generator=filter_generator)
data_selector = DataSelector(model=model)

# -------- load bar data ----------
bar_data_file_path = os.path.join(root_dir, "data_bar")
data_file_paths = [os.path.join(bar_data_file_path, "volume_sampled", "20170125_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170126_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170127_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170131_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170201_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170202_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170203_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170206_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170207_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170208_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170209_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170210_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170213_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170214_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170215_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170216_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170217_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170220_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170221_morning_volume_100_sampled.csv"),
                   os.path.join(bar_data_file_path, "volume_sampled", "20170222_morning_volume_100_sampled.csv")]

for data_file_path in data_file_paths:
    data_selector.append_data_set(pd.read_csv(data_file_path))

train_results, test_results = data_selector.sequential_cross_val(10, 3)

print("---- train results ----")
for train_result in train_results:
    print(train_result)

print("---- test results ----")
for test_result in test_results:
    print(test_result)
"""
bar_data_file_path = os.path.join(root_dir, "data_bar")
train_bar_file_paths = [os.path.join(bar_data_file_path, "volume_sampled", "20170125_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170126_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170127_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170131_morning_volume_100_sampled.csv")]

                        os.path.join(bar_data_file_path, "volume_sampled", "20170201_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170202_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170203_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170206_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170207_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170208_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170209_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170210_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170213_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170214_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170215_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170216_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170217_morning_volume_100_sampled.csv")]

test_bar_file_paths = [os.path.join(bar_data_file_path, "volume_sampled", "20170220_morning_volume_100_sampled.csv")]
                       #os.path.join(bar_data_file_path, "volume_sampled", "20170221_morning_volume_100_sampled.csv"),
                       #os.path.join(bar_data_file_path, "volume_sampled", "20170222_morning_volume_100_sampled.csv")]


for train_bar_file_path in train_bar_file_paths:
    model.append_train_data_set(pd.read_csv(train_bar_file_path))

for test_bar_file_path in test_bar_file_paths:
    model.append_test_data_set(pd.read_csv(test_bar_file_path))

# -------- run the model ---------
model.pre_process()
train_upper_metrics, train_lower_metrics = model.train()
test_upper_metrics, test_lower_metrics = model.test()

# ------- output --------
print(train_upper_metrics)
print(train_lower_metrics)
print(test_upper_metrics)
print(test_lower_metrics)

"""
