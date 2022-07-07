import datasets
import optuna
from datasets import load_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datasets import load_metric, ClassLabel

def main():
    #%%
    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train':'sentimentAnnotations_CSV/annotation_450.csv'}).shuffle()

    # Splits based off labels.
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])
    #%%
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels)

    def compute_metrics2(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="macro")
        precision = precision_score(y_true=labels, y_pred=pred, average="macro")
        f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    scores = {}
    # Override train/test
    for split in splits:
        splitx = split
    data = None
    data = datasets.load_dataset('csv', data_files={'train':'sentimentAnnotations_CSV/annotation_450.csv'})
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

    #%%
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
    #%%
    study = optuna.create_study(study_name='hyper-parameter-search', direction='minimize')
    study.optimize(func=objective, n_trials=15)
    print(study.best_value)
    print(study.best_params)
    print(study.best_trial)


if __name__ == "__main__":
    main()