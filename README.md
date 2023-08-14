# Rallypoint suicide detection

# Data

`./data/input/final_datasets/` datasets
```
├── Profile_features.csv (which posts belong to each military branch, etc.)
├── test.csv
├── train.csv
└── val.csv
```

`./data/output/`
  ```
  ├── performance
  │   ├── LGBM_23-08-04T16-59-04
  │   ├── LogReg_23-08-04T16-59-04
  │   ├── roberta-base_text_20230807-203851
  │   └── roberta_pretrained_meta_text_num(final)
  └── semantic_analysis
  ```


# Code

### Requirements

For all roberta multimodal models and TFIDF models
```
conda create -n rallypoint_stb_detector python=3.7 pandas numpy scikit-learn seaborn matplotlib jupyterlab redis==3.5.0 transformers==4.26.1 multimodal-transformers==0.2a0 
conda activate rallypoint_stb_detector
```

For roberta text model
```
conda create -n rallypoint_stb_detector_text python=3.10.12 pandas numpy scikit-learn matplotlib jupyterlab torch==2.0.1 datasets==2.14.3 transformers==4.28.1 accelerate==0.15.0 optuna==3.2.0
```

For semantic analysis
```
conda create -n rallypoint_stb_semantic python=3.10.12 pandas numpy scikit-learn seaborn==0.12.2 matplotlib==3.6.0 jupyterlab spacy==3.6.1 
conda activate rallypoint_stb_semantic
pip install scattertext==0.1.19 pytextrank==3.2.5 textalloc==0.0.3
python -m spacy download en_core_web_sm
```




### Models

`LogReg.tfidf.ipynb` Logistic Regression models

`LGBM_tfif.ipynb` Light GBM models

`roberta_text.ipynb` Finetuning, hyperparameter search, evaluation

Roberta text+metadata model
- `roberta_text_metadata_training.ipynb` Finetuning, hyperparameter search
- `roberta_text_metadata_evaluation.ipynb` Evaluation
- `roberta_text_metadata_training_minimal.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/rallypoint_suicide_detection/blob/main/multimodal_suicide_detector_minimal.ipynb)  Minimal version for deployment

### Statistical differences between types of posts 

`scatter_text.ipynb` words used in suicidal and nonsuicidal posts

`metadata_stats.ipynb` descriptive stats comparing metadata 

