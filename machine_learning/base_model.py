import pandas as pd
from machine_learning.feature_generators import FeatureGenerator
from machine_learning.filter_generators import FilterGenerator
from machine_learning.label_generators import LabelGenerator
import time
import pickle


class PreProcessor:
    def __init__(self,
                 feature_generator: FeatureGenerator,
                 label_generator: LabelGenerator,
                 filter_generator: FilterGenerator):
        self.feature_generator = feature_generator
        self.label_generator = label_generator
        self.filter_generator = filter_generator
        self.train_data_sets = []
        self.test_data_sets = []
        self.processed_train_df = pd.DataFrame()
        self.processed_test_df = pd.DataFrame()
        self.filter_col_name = self.filter_generator.get_filter_name()
        self.feature_list = self.feature_generator.get_feature_list()
        self.label_list = self.label_generator.get_label_list()

    def append_train_data_set(self, data_bar_df: pd.DataFrame) -> None:
        self.train_data_sets.append(data_bar_df)

    def append_test_data_set(self, data_bar_df: pd.DataFrame) -> None:
        self.test_data_sets.append(data_bar_df)

    def pre_process(self) -> None:
        start_time = time.time()
        print("---------- Preprocessing start --------------")
        # ----- START -----
        if len(self.train_data_sets) == 0:
            self.processed_train_df = pd.DataFrame()
        else :
            train_feature_list = [self.feature_generator.create_features(bar_df) for bar_df in self.train_data_sets]
            train_label_list = [self.label_generator.create_labels(bar_df) for bar_df in self.train_data_sets]
            train_filter_list = [self.filter_generator.create_filter(bar_df) for bar_df in self.train_data_sets]
            train_comb_list = [pd.concat([feature_df, label_df, filter_df], axis=1) for feature_df, label_df, filter_df
                               in zip(train_feature_list, train_label_list, train_filter_list)]
            self.processed_train_df = pd.concat(train_comb_list, ignore_index=True, axis=0)

        if len(self.test_data_sets) == 0:
            self.processed_test_df = pd.DataFrame()
        else :
            test_feature_list = [self.feature_generator.create_features(bar_df) for bar_df in self.test_data_sets]
            test_label_list = [self.label_generator.create_labels(bar_df) for bar_df in self.test_data_sets]
            test_filter_list = [self.filter_generator.create_filter(bar_df) for bar_df in self.test_data_sets]
            test_comb_list = [pd.concat([feature_df, label_df, filter_df], axis=1) for feature_df, label_df, filter_df
                              in zip(test_feature_list, test_label_list, test_filter_list)]
            self.processed_test_df = pd.concat(test_comb_list, ignore_index=True, axis=0)
        # ---- END -----
        end_time = time.time()
        print("---------- end : elapsed time : " + str(end_time - start_time) + "----------")

    def reset(self) -> None:
        self.train_data_sets = []
        self.test_data_sets = []

class BaseModel(PreProcessor):
    def __init__(self, feature_generator: FeatureGenerator, label_generator: LabelGenerator,
                 filter_generator: FilterGenerator):
        super().__init__(feature_generator, label_generator, filter_generator)

    def train(self) -> [dict]:
        if len(self.processed_train_df) == 0:
            raise Exception("there is no data to train on!")

    def test(self) -> [dict]:
        if len(self.processed_test_df) == 0:
            raise Exception("there is no data to test on!")

    def reset(self) -> None:
        super().reset()
        self.processed_train_df = pd.DataFrame()
        self.processed_test_df = pd.DataFrame()

    def remove_data_and_save(self, save_path):
        " uses pickle to save, removes data to avoid saving unnecessary data "
        self.train_data_sets = []
        self.test_data_sets = []
        pickle_out = open(save_path, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()



