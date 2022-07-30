import argparse
import datasets
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datasets import load_metric, ClassLabel

"""
Script for hyper parameter tuning a bert classification model

Required arguments:
    --train : PATH to train data-set file

Written by Christopher Bagdon 06/2022
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train':args.train}).shuffle()

    # Splits based off labels. Only use the last split. Messy hack to get a shuffled stratified train and dev set
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])
    for split in splits:
        splitx = split
    data = None
    data = datasets.load_dataset('csv', data_files={args.train})
    data['test'] = data['train'].select(splitx[1])
    data['train'] = data['train'].select(splitx[0])
    labels = ClassLabel(num_classes=3, names=['positive', 'negative', 'neutral'])

    def preprocess_function(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True)
        tokens['label'] = labels.str2int(batch['label'])
        return tokens
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_data = data.map(preprocess_function, batched=True,
                              remove_columns=data["train"].column_names
                              )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define trial
    def objective(trial: optuna.Trial):
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        training_args = TrainingArguments(output_dir='ade-test',
                                          learning_rate=trial.suggest_loguniform('learning_rate', low=4e-5, high=0.01),
                                          weight_decay=trial.suggest_loguniform('weight_decay', 4e-5, 0.01),
                                          num_train_epochs=trial.suggest_int('num_train_epochs', low = 2,high = 5),
                                        per_device_train_batch_size=8,
                                        per_device_eval_batch_size=8,
                                        disable_tqdm=True)
        trainer = Trainer(model=model,
                            args=training_args,
                            train_dataset=tokenized_data['train'],
                            eval_dataset=tokenized_data['test'],
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            )
        result = trainer.train()
        return result.training_loss

    # Run trial
    study = optuna.create_study(study_name='hyper-parameter-search', direction='minimize')
    study.optimize(func=objective, n_trials=15)
    # Print results
    print(study.best_value)
    print(study.best_params)
    print(study.best_trial)


if __name__ == "__main__":
    main()