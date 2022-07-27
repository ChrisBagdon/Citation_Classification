import tools
import argparse
import csv
import pandas as pd

"""
Script for training, and running a Naive bayes classification model.
Prints evaluation of test set

Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file

Written by Christopher Bagdon 06/2022
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--term_probs", required=False)
    parser.add_argument("--term_count", default=20, required=False)
    args = parser.parse_args()

    # Load training and test data
    with open(args.train) as train_file_text:
        train_data_text = csv.reader(train_file_text, delimiter="\t")
        X_train_text, Y_train_text = [], []
        for row in train_data_text:
            X_train_text.append(row[2])
            Y_train_text.append(row[3])

    with open(args.test) as dev_file_text:
        dev_data_text = csv.reader(dev_file_text, delimiter="\t")
        X_dev_text, Y_dev_text = [], []
        for row in dev_data_text:
            X_dev_text.append(row[2])
            Y_dev_text.append(row[3])

    # Instantiate and train text model
    text_model = tools.naive_bayes()
    text_model.train(X_train_text, Y_train_text)

    # Make predictions with text model
    predictions_text = text_model.predict(X_dev_text)

    # Get confusion matrix and scores of text model
    cf_text, scores_text = tools.evaluate(predictions_text, Y_dev_text)
    print(scores_text)


    if args.term_probs:
        # Gather term probabilites from model
        background_text = pd.DataFrame(text_model.labels['background']['term_probs'].items()).sort_values(by=[1],
                                                                                                          ascending=False).reset_index()
        result_text = pd.DataFrame(text_model.labels['result']['term_probs'].items()).sort_values(by=[1],
                                                                                                  ascending=False).reset_index()
        method_text = pd.DataFrame(text_model.labels['method']['term_probs'].items()).sort_values(by=[1],
                                                                                                  ascending=False).reset_index()
        # Restrict to top X terms per class
        count = args.term_count
        term_prob_text = pd.DataFrame(
            [background_text[0][:count], background_text[1][:count], result_text[0][:count], result_text[1][:count],
             method_text[0][:count], method_text[1][:count], ]).transpose()
        print(term_prob_text)

if __name__ == "__main__":
    main()