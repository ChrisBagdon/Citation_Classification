import datasets
import gc
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets.features.features import ClassLabel
import numpy as np
from datasets import load_metric
from tools import evaluate
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import argparse

"""
Script for training, saving, and running a transformer-based sentiment classification model.
Prints evaluation of test set

Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file
    --save  : PATH to save/load model 
Optional arguments:
    --model : Name of base model to be used for embeddings and base-model
    --lr    : Learning rate
    --epochs: Number of epochs
    --decay : Weight decay

Written by Christopher Bagdon 06/2022
"""

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--model", default="distilbert-base-uncased")
parser.add_argument("--lr", default=1e-5)
parser.add_argument("--epochs", default=5)
parser.add_argument("--decay", default=0.01)

args = parser.parse_args()
# Clear cuda memory
gc.collect()
torch.cuda.empty_cache()
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    print("GPU Usage after emptying the cache")
    gpu_usage()
free_gpu_cache()


# Helper methods for trainer metrics
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
def compute_metrics2(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Train and run cross-validation experiment
def cross_validate_sent():
    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train': 'sentimentAnnotations_CSV/annotation_450.csv'}).shuffle()

    # Splits based off labels.
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])

    scores = {}
    # Run model for each split
    for i, split in enumerate(splits):
        # Reset data
        data = None
        # Load split
        data = datasets.load_dataset('csv', data_files={'train': args.train})
        data['test'] = data['train'].select(split[1])
        data['train'] = data['train'].select(split[0])
        labels = ClassLabel(num_classes=3, names=['positive', 'negative', 'neutral'])
        # Preprocess function for tokenizer
        def preprocess_function(batch):
            tokens = tokenizer(batch['text'], padding='max_length', truncation=True)
            tokens['label'] = labels.str2int(batch['label'])
            return tokens
        # Create tokenizer with model from script args
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        # Reset data and tokenize split
        tokenized_data = None
        tokenized_data = data.map(preprocess_function, batched=True)
        # Load data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Load model from script args
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        # Set model output dir based on split
        output_dir = args.model + "_" + str(i)
        # Set trainer args and initialize trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=args.epochs,
            weight_decay=args.decay,
            gradient_accumulation_steps=8,
            # remove_unused_columns=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics2
        )
        # Fine-tune model and save
        trainer.train()
        trainer.save_model(output_dir + "/model")
        """for file in os.listdir(output_dir):
            if file.startswith("checkpoint"):
                model_dir = os.path.join(output_dir,file)"""
        # Build pipeline from model
        classifier = pipeline(task='text-classification', model=output_dir + "/model", tokenizer=tokenizer)
        # Grab test set of split for predictions
        X_dev, Y_dev = data["test"]['text'], data["test"]['label']
        predictions = classifier(X_dev)
        # COnvert output to labels
        labels = {'LABEL_0': 'positive', 'LABEL_1': 'negative', 'LABEL_2': 'neutral'}
        preds = [labels[x['label']] for x in predictions]
        # Save scores for compiling after all splits finish
        cf, score = evaluate(preds, Y_dev)
        scores[i] = score
        # score.to_pickle("cross_val_scores_"+str(i)+".pkl")
    # Create dataframe of all splits' scores
    df_concat = pd.concat(scores.values())
    # Compute mean of splits
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    print(df_means)
    df_means.to_csv('means_450.csv')

# Create model from single split
def save_sent_model():
    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train': 'sentimentAnnotations_CSV/annotation_450.csv'}).shuffle()

    # Splits based off labels.
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])
    for split in splits:
        splitx = split
    data = datasets.load_dataset('csv', data_files={'train':'sentimentAnnotations_CSV/annotation_450.csv'})
    data['test'] = data['train'].select(splitx[1])
    data['train'] = data['train'].select(splitx[0])
    labels = ClassLabel(num_classes=3, names=['positive', 'negative', 'neutral'])

    # Preprocess function for tokenizer
    def preprocess_function(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True)
        tokens['label'] = labels.str2int(batch['label'])
        return tokens
    # Create tokenizer with model from script args
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_data = data.map(preprocess_function, batched=True)
    # Load data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Load model from script args
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    training_args = TrainingArguments(
        output_dir=args.model+"_tuned",
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        gradient_accumulation_steps=8,
        # remove_unused_columns=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics2, )
    # Fine-tune model and save
    trainer.train()
    trainer.save_model(args.save)

# Load saved model and run on given dataset
def classify_sent(data, model_path):
    train_dataset = datasets.load_dataset('csv',
                                          data_files={'train': args.data})
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Load saved model
    classifier = pipeline(task='text-classification', model=model_path, tokenizer=tokenizer)
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    # Run data through model
    predictions = classifier(data, **tokenizer_kwargs)
    # Convert and return prediction labels
    labels = {'LABEL_0': 'positive', 'LABEL_1': 'negative', 'LABEL_2': 'neutral'}
    return [labels[x['label']] for x in predictions]

def main():


    run_model = False
    path = "Sentiment_Full_Model"


    if args.cross:
        cross_validate_sent()

    if args.train:
        save_sent_model()

    if args.classify:
        classify_sent()

if __name__ == "__main__":
    main()