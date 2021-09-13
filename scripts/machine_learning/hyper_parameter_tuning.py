import definitions
import os
import pandas as pd
from machine_learning.feature_generators import TaLibGenerator
from machine_learning.label_generators import DoubleBarrierLabel
from machine_learning.filter_generators import EveryKthFilter
from machine_learning.model_selector import HyperParameterTuner
from sklearn.ensemble import RandomForestClassifier

root_dir = definitions.ROOTDIR
# ----- inputs ----
look_ahead = 15
upper_barrier = 10
lower_barrier = 10
filter_period = 1

grid = {
    "n_estimators": [150, 200, 250],
    "max_depth": [4, 7, 10],
    "min_samples_leaf": [5, 10, 15],
    "min_samples_split": [5, 10, 15],
    "max_features": [4, 6, 8],
    "random_state": [4]
}

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

base_estimator = RandomForestClassifier()
feature_generator = TaLibGenerator(atr_time_period=atr_time_period, adx_time_period=adx_time_period,
                                   apo_fast_period=apo_fast_period,
                                   rsi_time_period=rsi_time_period, roc_time_period=roc_time_period,
                                   ult_osc_period_1=ult_osc_period_1, ult_osc_period_2=ult_osc_period_2,
                                   tsf_time_period=tsf_time_period, p_corr_time_period=p_corr_time_period,
                                   macd_fast_period=macd_fast_period, macd_slow_period=macd_slow_period,
                                   macd_signal_period=macd_signal_period)
label_generator = DoubleBarrierLabel(look_ahead=look_ahead, upper_barrier=upper_barrier, lower_barrier=lower_barrier)
filter_generator = EveryKthFilter(period=filter_period)
hyper_param_tuner = HyperParameterTuner(feature_generator=feature_generator, label_generator=label_generator,
                                        filter_generator=filter_generator)

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

for data_file_path in data_file_paths[:10]:
    hyper_param_tuner.append_train_data_set(pd.read_csv(data_file_path))

for data_file_path in data_file_paths[10:13]:
    hyper_param_tuner.append_test_data_set(pd.read_csv(data_file_path))

hyper_param_tuner.pre_process()
best_scores, best_grids = hyper_param_tuner.grid_search(base_estimator = base_estimator, grid = grid)
print(best_scores)
print(best_grids)
