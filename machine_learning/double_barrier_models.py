import pandas as pd
from machine_learning.feature_generators import FeatureGenerator
from machine_learning.label_generators import DoubleBarrierLabel
from machine_learning.filter_generators import FilterGenerator
from machine_learning.base_model import BaseModel
import copy
from machine_learning.model_performance import binary_classification_metrics
import time


class DoubleBarrierClassificationModel(BaseModel):
    """
    This model does classification
    This model assumes a double barrier labeling scheme
    """
    def __init__(self, feature_generator: FeatureGenerator, label_generator: DoubleBarrierLabel,
                 filter_generator: FilterGenerator, upper_estimator, lower_estimator):
        super().__init__(feature_generator, label_generator, filter_generator)
        self.empty_upper_model = copy.deepcopy(upper_estimator)
        self.empty_lower_model = copy.deepcopy(lower_estimator)
        self.upper_model = copy.deepcopy(upper_estimator)
        self.lower_model = copy.deepcopy(lower_estimator)

        self.look_ahead = label_generator.look_ahead
        self.upper_barrier = label_generator.upper_barrier
        self.lower_barrier = label_generator.lower_barrier

        self.upper_barrier_col_name = self.label_generator.get_label_list()[0]
        self.lower_barrier_col_name = self.label_generator.get_label_list()[1]

    def append_train_data_set(self, data_bar_df : pd.DataFrame) -> None:
        self.train_data_sets.append(data_bar_df)

    def append_test_data_set(self, data_bar_df : pd.DataFrame) -> None:
        self.test_data_sets.append(data_bar_df)

    def train(self) -> [dict]:
        start_time = time.time()
        print("---------- Training start --------------")
        # ----- START -----
        train_df = self.processed_train_df[self.processed_train_df[self.filter_col_name]].dropna()
        train_features = train_df[self.feature_list]
        y_train_upper = train_df[self.upper_barrier_col_name]
        y_train_lower = train_df[self.lower_barrier_col_name]
        self.upper_model.fit(train_features, y_train_upper)
        self.lower_model.fit(train_features, y_train_lower)
        y_pred_upper = self.upper_model.predict(train_features)
        y_pred_lower = self.lower_model.predict(train_features)
        # ---- END -----
        end_time = time.time()
        print("---------- end : elapsed time : " + str(end_time - start_time) + "----------")
        return binary_classification_metrics(y_train_upper, y_pred_upper), binary_classification_metrics(y_train_lower, y_pred_lower)

    def test(self) -> [dict]:
        start_time = time.time()
        print("---------- Testing start --------------")
        # ----- START -----
        test_df = self.processed_test_df[self.processed_test_df[self.filter_col_name]].dropna()
        test_features = test_df[self.feature_list]
        y_test_upper = test_df[self.upper_barrier_col_name]
        y_test_lower = test_df[self.lower_barrier_col_name]
        y_pred_upper = self.upper_model.predict(test_features)
        y_pred_lower = self.lower_model.predict(test_features)
        # ---- END -----
        end_time = time.time()
        print("---------- end : elapsed time : " + str(end_time - start_time) + "----------")
        return binary_classification_metrics(y_test_upper, y_pred_upper), binary_classification_metrics(y_test_lower, y_pred_lower)

    def get_params(self) -> (int, float, float):
        return self.look_ahead, self.upper_barrier, self.lower_barrier

    def reset(self) -> None:
        super().reset()
        self.upper_model = copy.deepcopy(self.empty_upper_model)
        self.lower_model = copy.deepcopy(self.empty_lower_model)

    def predict(self, bar_df):
        features = self.feature_generator.create_features(bar_df)
        labels = self.label_generator.create_labels(bar_df)
        filter_ser = self.filter_generator.create_filter(bar_df)
        bar_pred_df = pd.concat([features, labels, filter_ser], axis = 1)
        bar_pred_df = bar_pred_df[bar_pred_df[self.filter_generator.get_filter_name()]].dropna()
        bar_pred_df["upper_barrier_prediction"] = self.upper_model.predict(bar_pred_df[self.feature_list])
        bar_pred_df["lower_barrier_prediction"] = self.lower_model.predict(bar_pred_df[self.feature_list])
        return bar_pred_df












