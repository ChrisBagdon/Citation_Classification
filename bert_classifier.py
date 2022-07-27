import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding,\
    AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    pipeline
from datasets.features.features import ClassLabel
from datasets import load_metric
import argparse
import csv
import numpy as np
from tools import evaluate


"""
Script for training, saving, and running a transformer-based classification model.
Prints evaluation of test set

Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file
Optional arguments:
    --model : Name of base model to be used for embeddings and base-model
    --lr    : Learning rate
    --epochs: Number of epochs
    --decay : Weight decay
    
Written by Christopher Bagdon 06/2022
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--decay", default=0.01)


    args = parser.parse_args()

    # load train and test sets into datasets object
    data = datasets.load_dataset('csv', data_files={
                                        'train': args.train,
                                        'test': args.test})
    data = data.rename_columns({'Column1': 'ID', 'Column2': 'exp',
                                'Column3': 'text', 'Column4': 'label'})
    data = data.remove_columns(['exp', 'ID'])

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set labels
    labels = ClassLabel(num_classes=3, names=['background', 'result', 'method'])

    # Preprocessing function using tokenizer
    def preprocess_function(batch):
        tokens = tokenizer(batch['text'], padding=True, truncation=True, max_length=128)
        tokens['label'] = labels.str2int(batch['label'])
        return tokens

    # Tokenize data
    tokenized_data = data.map(preprocess_function, batched=True)

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define metric for trainer
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    # Set training args and initialize trainer
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        remove_unused_columns=True,
        evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    # Fine-tune model and save
    trainer.train()
    trainer.save_model(args.model+"fine-tuned")

    # Create pipeline with fine-tuned model
    classifier = pipeline(task='text-classification',
                          model=args.model+"fine-tuned",
                          tokenizer=tokenizer)
    # Load test set (Not at datasets object)
    with open(args.test) as dev_file:
        dev_data = csv.reader(dev_file, delimiter="\t")
        X_dev, Y_dev = [], []
        for row in dev_data:
            X_dev.append(row[2])
            Y_dev.append(row[3])

    # Run test set through model
    predictions = classifier(X_dev)

    # Convert output to labels
    labels = {'LABEL_0': 'background', 'LABEL_1': 'result', 'LABEL_2': 'method'}
    preds = [labels[x['label']] for x in predictions]

    # Evaluate results
    cf, scores = evaluate(preds, Y_dev)
    print(scores)
    print(cf)



if __name__ == "__main__":
    main()