import datasets
import gc
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets.features.features import ClassLabel
import numpy as np
from datasets import load_metric
from tools import evaluate
from sklearn.model_selection import StratifiedKFold
from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function
import sklearn
import GPy
import numpy as np
import pandas as pd
from opendelta import BitFitModel, LoraModel
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
import os
from transformers import AutoConfig, MAMConfig
def main():
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


    # Make the kfold object
    folds = StratifiedKFold(n_splits=5)
    # Then get the dataset
    data = datasets.load_dataset('csv', data_files={'train':'sentimentAnnotations_CSV/annotation_450.csv'}).shuffle()

    # Splits based off labels.
    splits = folds.split(np.zeros(data["train"].num_rows), data["train"]["label"])

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
    tokenized_data = data.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=3)
    output_dir = "sent-hyper-distil"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
        #remove_unused_columns=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
        )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics2,)

    tune_config = {
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": tune.choice([5,10,15]),
        "max_steps": -1,
    }
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-1, 5e-6),
            "per_device_train_batch_size": [8],
        },
    )


    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_loss", "epoch", "training_iteration"],
    )
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=10,
        resources_per_trial={"cpu": 1, "gpu": 1},
        scheduler=pbt,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop=None,
        progress_reporter=reporter,
        local_dir="ray_results/",
        name=output_dir+"_search",
        log_to_file=True,
    )
    result = pd.DataFrame(best_run.hyperparameters, index=[0])
    result['acc'] = best_run.objective
    print(result)
    result.to_csv("Hyper_search_result.csv", mode='a', index=False)





if __name__ == "__main__":
    main()