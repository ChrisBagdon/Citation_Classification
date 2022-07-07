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


def compute_metrics2(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def cross_validate_sent():
    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train': 'sentimentAnnotations_CSV/annotation_450.csv'}).shuffle()

    # Splits based off labels.
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels)

    scores = {}
    # Override train/test
    for i, split in enumerate(splits):
        data = None
        data = datasets.load_dataset('csv', data_files={'train': 'sentimentAnnotations_CSV/annotation_450.csv'})
        data['test'] = data['train'].select(split[1])
        data['train'] = data['train'].select(split[0])
        labels = ClassLabel(num_classes=3, names=['positive', 'negative', 'neutral'])

        def preprocess_function(batch):
            tokens = tokenizer(batch['text'], padding='max_length', truncation=True)
            tokens['label'] = labels.str2int(batch['label'])
            return tokens

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenized_data = None
        tokenized_data = data.map(preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        output_dir = "sent-results-distilbert_" + str(i)
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=0.00011715503310767902,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=4,
            weight_decay=5.399698478570332e-05,
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

        trainer.train()
        trainer.save_model(output_dir + "/model")
        """for file in os.listdir(output_dir):
            if file.startswith("checkpoint"):
                model_dir = os.path.join(output_dir,file)"""
        classifier = pipeline(task='text-classification', model=output_dir + "/model", tokenizer=tokenizer)

        X_dev, Y_dev = data["test"]['text'], data["test"]['label']
        predictions = classifier(X_dev)
        labels = {'LABEL_0': 'positive', 'LABEL_1': 'negative', 'LABEL_2': 'neutral'}
        preds = [labels[x['label']] for x in predictions]

        cf, score = evaluate(preds, Y_dev)
        scores[i] = score
        # score.to_pickle("cross_val_scores_"+str(i)+".pkl")
    df_concat = pd.concat(scores.values())
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    print(df_means)
    df_means.to_csv('means.csv')

def save_sent_model(output_dir):
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

    def preprocess_function(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True)
        tokens['label'] = labels.str2int(batch['label'])
        return tokens

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=0.00011715503310767902,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        weight_decay=5.399698478570332e-05,
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

    trainer.train()
    trainer.save_model(output_dir + "/model")


def classify_sent(data, model_path):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifier = pipeline(task='text-classification', model=model_path, tokenizer=tokenizer)
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    predictions = classifier(data, **tokenizer_kwargs)
    labels = {'LABEL_0': 'positive', 'LABEL_1': 'negative', 'LABEL_2': 'neutral'}
    return [labels[x['label']] for x in predictions]

def main():

    cross_validate = False

    run_model = True
    path = "Sentiment_Full_Model"


    if cross_validate:
        cross_validate_sent()

    if run_model:
        save_sent_model(path)

if __name__ == "__main__":
    main()