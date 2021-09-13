from machine_learning.base_model import BaseModel
from machine_learning.base_model import PreProcessor
import pandas as pd
from machine_learning.feature_generators import FeatureGenerator
from machine_learning.filter_generators import FilterGenerator
from machine_learning.label_generators import LabelGenerator
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score
import copy

class DataSelector:
    """ Makes it easier to work through different combinations of train and test sets given a master data set
    Assumes that the datasets given to this object are 'equal', Respects the ordering """
    def __init__(self, model : BaseModel):
        self.model = model
        self.data_sets = []
        self.num_data_sets = 0

    def append_data_set(self, bar_df : pd.DataFrame):
        self.data_sets.append(bar_df)
        self.num_data_sets = len(self.data_sets)

    def sequential_cross_val(self, num_train_sets : int, num_test_sets : int):
        self.model.reset()
        train_results = []
        test_results = []
        for ptr in range(self.num_data_sets - num_train_sets - num_test_sets + 1):
            for ptr2 in range(ptr, ptr + num_train_sets):
                self.model.append_train_data_set(self.data_sets[ptr2])
            for ptr2 in range(ptr + num_train_sets, ptr + num_train_sets + num_test_sets):
                self.model.append_test_data_set(self.data_sets[ptr2])
            self.model.pre_process()
            train_results.append(self.model.train())
            test_results.append(self.model.test())
        return train_results, test_results


class HyperParameterTuner(PreProcessor):
    def __init__(self, feature_generator: FeatureGenerator, label_generator: LabelGenerator,
                 filter_generator: FilterGenerator):
        super().__init__(feature_generator, label_generator, filter_generator)

    def grid_search(self, base_estimator, grid : dict) -> (dict, dict):
        """ applies sklearn grid_search_CV to each label, currently uses precision as a scoring method"""
        estimator = copy.deepcopy(base_estimator)
        best_grids = {}
        best_scores = {}
        for label in self.label_list:
            train_df = self.processed_train_df[self.processed_train_df[self.filter_col_name]].dropna()
            train_features = train_df[self.feature_list]
            train_target = train_df[label]
            test_df = self.processed_test_df[self.processed_test_df[self.filter_col_name]].dropna()
            test_features = test_df[self.feature_list]
            test_target = test_df[label]
            best_grid = None
            best_score = 0
            for g in ParameterGrid(grid):
                estimator.set_params(**g)
                estimator.fit(train_features, train_target)
                test_prediction = estimator.predict(test_features)
                score = precision_score(test_target, test_prediction)
                if score > best_score:
                    best_score = score
                    best_grid = g
            best_scores[label] = best_score
            best_grids[label] = best_grid
        return best_scores, best_grids









