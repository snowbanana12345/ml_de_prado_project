import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def binary_classification_metrics(true_values : np.array, predicted_values : np.array) -> dict:
    score_dict = {}
    score_dict["accuracy_score"] = accuracy_score(true_values, predicted_values)
    conf_mat = confusion_matrix(true_values, predicted_values)
    score_dict["true_positive"] = conf_mat[1][1]
    score_dict["false_positive"] = conf_mat[0][1]
    score_dict["true_negative"] = conf_mat[0][0]
    score_dict["false_negative"] = conf_mat[1][0]
    return score_dict




