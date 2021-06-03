import numpy as np

#Metrics
def recall(y_test, y_pred):

    true_positives = np.sum(np.round(np.clip(y_test * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(float).eps)
    return recall

def precision(y_test, y_pred):

    true_positives = np.sum(np.round(np.clip(y_test * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(float).eps)
    return precision

def f1_score(y_test, y_pred):

    test_np = y_test.numpy()
    pred_np = y_pred.numpy()
    precision_score = precision(test_np, pred_np)
    recall_score = recall(test_np, pred_np)
    return 2*((precision_score * recall_score) / (precision_score + recall_score + np.finfo(float).eps))


