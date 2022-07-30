# Citation_Classification
Group project for Stuttgart Uni Team Lab Course

Work was initially done and tested in notebooks before porting them to python scripts. Please use scripts to run experiments and only refer to the notebooks as reference to the work history.

Naive Bayes:
Script: nb_classifer.py
Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file

Bert Classifier:
Script: bert_classifer.py
Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file
Optional arguments:
    --model : Name of base model to be used for embeddings and base-model
    --lr    : Learning rate
    --epochs: Number of epochs
    --decay : Weight decay

Sentiment Classifier:
Script: sentiment_classifier.py
Required arguments:
    --train : PATH to train data-set file
    --test  : PATH to test data-set file
    --save  : PATH to save/load model 
Optional arguments:
    --model : Name of base model to be used for embeddings and base-model
    --lr    : Learning rate
    --epochs: Number of epochs
    --decay : Weight decay

Hyper-parameter tuning:
Script: optuna_search.py
Required arguments:
    --train : PATH to train data-set file