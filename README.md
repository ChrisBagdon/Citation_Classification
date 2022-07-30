# Citation_Classification
Group project for Stuttgart Uni Team Lab Course

Work was initially done and tested in notebooks before porting them to Python scripts. Please use scripts to run experiments and only refer to the notebooks as reference to the work history. The various notebooks are labeled and commented for readability. The dataExploration.ipynb and sentimentAnnotations.ipynb notebooks include relevant work and visualizations outside of the main Python scripts for reference.

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
    
dataExploration.ipynb:
Initial data exploration during the early phases of our project.
Includes visualizations as we explored the distribution of the SciCite corpus.

sentimentAnnotations.ipynb:
Displays work for creating CSVs with our sentiment annotations.
Inter-annotator agreement score (Cohen's kappa): 0.73
