from math import log

import pandas as pd
import numpy as np
import jsonlines

### Import dataset ###
def import_json(path):
    train_jsonl_list = []
    with jsonlines.open(path) as f:
        for line in f.iter():
            train_jsonl_list.append(line)
    return pd.DataFrame(train_jsonl_list)

### Evaluation Methods ###
def evaluate(predictions, gold_standard):
    # Collect all unique labels from predictions and gold_std
    labels_set = set(predictions + gold_standard)
    labels = {}
    for i, label in enumerate(labels_set):
        labels[label] = i
    # Create confusion matrix
    confusion_matrix = np.zeros((len(labels_set),len(labels_set)))
    for pred, gold in zip(predictions, gold_standard):
        confusion_matrix[labels[pred]][labels[gold]] += 1
    labels_index = list(labels_set); labels_index.append('overall')
    columns = []
    # Create scores table
    scores = pd.DataFrame(np.zeros((len(labels_set), 3)))
    scores.columns = ['Precision', 'Recall', 'F1']
    overall_TP = 0
    # Calculate P, R, F1 and populate scores table
    for label in labels_set:
        i = labels[label]
        # Possible error case (Precision): denominator == 0; divide by 0
        if np.sum(confusion_matrix, axis=0)[i] == 0:
            scores['Precision'][i] = 0
        else:
            scores['Precision'][i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
        # Possible error case (Recall): denominator == 0; divide by 0
        if np.sum(confusion_matrix, axis=1)[i] == 0:
            scores['Recall'][i] = 0
        else:
            scores['Recall'][i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
        # Possible error case: P == 0 == R; divide by 0
        if scores['Precision'][i] == 0 and scores['Recall'][i] == 0:
            scores['F1'][i] = 0
        else:
            scores['F1'][i] = 2 * (scores['Precision'][i]*scores['Recall'][i]/(scores['Precision'][i]+scores['Recall'][i]))
        overall_TP += confusion_matrix[i][i]
    scores.loc[len(labels_set)] = [overall_TP / np.sum(confusion_matrix)] * 3
    scores.index = labels_index
    return (confusion_matrix, scores)

### Saves a scores DataFrame labeled by which model and feature values were used as a CSV
# Inputs: scores DataFrame, model_name string, feature_names list
# Outputs: model_name_feature_names CSV file
def save_scores(scores, model_name, feature_names):
    # Create full filepath and pass to df.to_csv
    output_directory = 'output_files_CSV/'
    filename = model_name
    if type(feature_names) == list:
        for feature in feature_names:
            filename += '_' + feature
    else:
        filename += '_' + feature_names
    filepath = output_directory + filename + '.csv'
    scores.to_csv(filepath)

### Model Methods ###

def tokenize(text, tokens):
  cur_token =""
  for char in text:
    # Check if is apart of token
    if char.isalnum() or char == "'":
      cur_token += char
      continue
    # Check for space
    elif char == " ":
      if len(cur_token) > 0:
        if cur_token in tokens:
          tokens[cur_token] += 1
          cur_token = ""
          continue
        else:
          tokens[cur_token] = 1
          cur_token = ""
          continue
    # Check if punctuation
    else:
      if len(cur_token) > 0:
        if cur_token in tokens:
          tokens[cur_token] += 1
          cur_token = ""
        else:
          tokens[cur_token] = 1
          cur_token = ""
      if char in tokens:
        tokens[char] += 1
        continue
      else:
        tokens[char] = 1
        continue
  if len(cur_token) > 0:
    if cur_token in tokens:
          tokens[cur_token] += 1
          cur_token = ""
    else:
      tokens[cur_token] = 1
      cur_token = ""


class naive_bayes:
    def __init__(self):
        self.labels = {}
        self.doc_count = 0
        self.bin_size = 0

    def train(self, X, Y):
        for string, label in zip(X, Y):
            # Count instances of labels
            if label not in self.labels:
                self.labels[label] = {'count': 1, 'terms': {}}
            else:
                self.labels[label]['count'] += 1
            # Count tokens from document
            tokenize(string, self.labels[label]['terms'])
            # Increase total document count
            self.doc_count += 1

        # Tally bin_size for smoothing
        terms_list = []
        for label, labels_dic in self.labels.items():
            terms_list = terms_list + list(labels_dic['terms'].keys())
        self.bin_size += len(set(terms_list))
        # Calculate class statistics
        for label, labels_dic in self.labels.items():
            # Calculate label prior probability
            self.labels[label]["prior"] = labels_dic['count'] / self.doc_count
            # Save total number of tokens in label + smoothing
            self.labels[label]["term_count"] = sum(labels_dic['terms'].values()) + self.bin_size
            # Calculate probability of each token in label + smoothing
            terms = labels_dic['terms'].keys()
            # for term in terms:
            # print((labels_dic['terms'][term]+1) / labels_dic["term_count"])
            self.labels[label]["term_probs"] = {term: (labels_dic['terms'][term] + 1) /
                                                      labels_dic["term_count"]
                                                for term in terms}

    def predict(self, X, use_log=True):
        predictions = []

        for string in X:
            tokens = {}
            tokenize(string, tokens)
            probabilities = []
            for label, label_dic in self.labels.items():
                if use_log:
                    prob = sum(log(label_dic["term_probs"][token]) * count
                               if token in label_dic['terms']
                               else log(1 / label_dic['term_count']) * count
                               for token, count in tokens.items()) \
                           + log(label_dic['prior'])
                else:
                    prob = label_dic['prior']
                    for token, count in tokens.items():
                        if token in label_dic['terms'].keys():
                            prob = prob * (label_dic["term_probs"][token] ** count)
                        else:
                            prob = prob * ((1 / label_dic['term_count']) ** count)
                            # print(prob)
                probabilities.append((label, prob))
            predictions.append(max(probabilities, key=lambda item: item[1])[0])

        return predictions



