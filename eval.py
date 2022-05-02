import pandas as pd
import numpy as np

def evaluate(predictions, gold_standard):
  labels_set = set(gold_standard)
  labels = {}
  for label, i in enumerate(labels_set):
    labels[label] = i
  confusion_matrix = np.zeros((len(labels),len(labels)))
  for pred, gold in zip(predictions, gold_standard):
    confusion_matrix[labels[pred]][labels[gold]] += 1

  columns = ["Precision", "Recall", "F1-Score"]
  scores = pd.DataFrame(np.zeros((len(labels) + 1, 3)), columns=columns)
  overall_TP = 0
  for i in len(labels):
    precision = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
    recall = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
    fScore = (2 * precision * recall) / (precision + recall)
    label_scores = [precision, recall, fScore]
    scores.iloc[i] = label_scores
    overall_TP += confusion_matrix[i][i]
  scores.iloc[len(labels)] = [overall_TP / np.sum(confusion_matrix)] * 3
    