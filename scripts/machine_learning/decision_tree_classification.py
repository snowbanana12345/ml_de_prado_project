import definitions
import os
import pandas as pd
import machine_learning.feature_engineering as fe
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ----- inputs ----
look_ahead = 8
threshold = 10
features = ["atr", "adx", "apo", "rsi", "roc", "ult_osc"]
label = "delta_label"
proba_threshold = 0.5

tree_depth = 10
tree_min_sample_leaf = 30
tree_min_sample_split = 30

# ----- load data -----
root_dir = definitions.ROOTDIR
bar_data_file_path = os.path.join(root_dir, "data_bar")
train_bar_file_paths = [os.path.join(bar_data_file_path, "volume_sampled", "20170125_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170126_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170127_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170131_morning_volume_100_sampled.csv"),
                        os.path.join(bar_data_file_path, "volume_sampled", "20170201_morning_volume_100_sampled.csv")]
test_bar_file_paths = [os.path.join(bar_data_file_path, "volume_sampled", "20170202_morning_volume_100_sampled.csv")]

train_bar_df_lst = [pd.read_csv(f_path) for f_path in train_bar_file_paths]
test_bar_df_lst = [pd.read_csv(f_path) for f_path in test_bar_file_paths]

#### ------ PREPROCESSING ------ ######
# ----- create features ------
train_bar_df_lst = [fe.create_feature_set_1(train_bar_df) for train_bar_df in train_bar_df_lst]
test_bar_df_lst = [fe.create_feature_set_1(test_bar_df) for test_bar_df in test_bar_df_lst]

# ----- create labels -----
train_bar_df_lst = [fe.create_delta_labels(train_bar_df, look_ahead, threshold) for train_bar_df in train_bar_df_lst]
test_bar_df = [fe.create_delta_labels(test_bar_df, look_ahead, threshold) for test_bar_df in test_bar_df_lst]

comb_train_bar_df = pd.concat(train_bar_df_lst, ignore_index = True, axis = 0)
comb_test_bar_df = pd.concat(test_bar_df_lst, ignore_index = True, axis = 0)

X_train = comb_train_bar_df[features]
y_train = comb_train_bar_df[label]
X_test = comb_test_bar_df[features]
y_test = comb_test_bar_df[label]

# ---- train the model ----
model = DecisionTreeClassifier(max_depth = tree_depth, min_samples_leaf = tree_min_sample_leaf, min_samples_split = tree_min_sample_split)
model.fit(X_train, y_train)

# ---- validation -----
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

train_accept_array = [max(proba) > proba_threshold for proba in y_train_proba]
test_accept_array = [max(proba) > proba_threshold for proba in y_test_proba]

y_train_pred_filtered = y_train_pred[train_accept_array]
y_train_filtered = y_train[train_accept_array]
y_test_pred_filtered = y_test_pred[test_accept_array]
y_test_filtered = y_test[test_accept_array]

train_accuracy = accuracy_score(y_train_filtered, y_train_pred_filtered)
test_accuracy = accuracy_score(y_test_filtered, y_test_pred_filtered)

train_conf_mat = confusion_matrix(y_train_filtered, y_train_pred_filtered)
test_conf_mat = confusion_matrix(y_test_filtered, y_test_pred_filtered)

print("train accuracy : " + str(train_accuracy))
print("test accuracy : " + str(test_accuracy))
print("train confusion matrix")
print(train_conf_mat)
print("test confusion matrix")
print(test_conf_mat)






