# Rallypoint suicide detection

# Slides

- Daniel on choosing threshold [here](https://docs.google.com/presentation/d/1gZgHj4qIQ9BmXwfV9K_Fel-1QlsSOQg8zzykpGiGZeM/edit?usp=sharing)

- Richard on final results and error analysis [here](https://docs.google.com/presentation/d/1HLiLwjkF9ryofq5LuyrZaupyq_rdKDIW/edit?usp=sharing&ouid=103429979135230884916&rtpof=true&sd=true)

# Data

Noah is hosting the model's data on Dropbox: 
 https://www.dropbox.com/sh/1clfl1u2v6p225y/AADDDIiLo3DDt3h_NG0zGzLfa?dl=0

 


`data/input/`
  - `rp/` datasets
  - `roberta_pretrained_meta_text_num(final)/pred_seed_2/` checkpoints

Some more data as saved in the Dropbox folder:
```
RallyPoint Milestone 6 Code
--Dataset Notebook
--Model Notebook
  --Multimodal-Toolkit
    --datasets
      -rp
        train.csv
        val.csv
        test.csv
    --logs
      --roberta_base_text(final)
      --roberta_pretrained_meta_text_num(final)
      --roberta_pretrained_meta_text(final)
        --pred_seed
        --preds
        --pred_seed_2
          --rp_config(meta).json
      --roberta_pretrained_text(final)
  --Compute Evaluation plots and Examples.ipynb
  --Hyperparameter_Tuning_of_Baseline_RP_Models.ipynb
  --Multi-modal Toolkit
  --Rallypoint_Multimodal_model_notebook(final).ipynb
  --Rallypoint_Multimodal_model_notebook.ipynb
  --baseline_models/
    
```
  
`data/output/`



# Code

virtual environment: `conda activate rallypoint_stb_detector`

`conda create -n rallypoint_stb_detector python=3.7 pandas numpy scikit-learn seaborn matplotlib jupyterlab`
`conda install 

This uses:
```
Python 3.7.12
redis==3.5.0
transformers==4.26.1 
multimodal-transformers==0.2a0
```

TODO 

`conda activate rallypoint_suicide_detection`
`requirements.txt`


`LGBM Hyperparameter Tuning Notebook.ipynb` LGBM models


`multimodal_suicide_detector_minimal.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/rallypoint_suicide_detection/blob/main/multimodal_suicide_detector_minimal.ipynb)
  - Minimal version for deployment.

`multimodal_suicide_detector.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/rallypoint_suicide_detection/blob/main/multimodal_suicide_detector.ipynb)
  - We ran these models in google colab to use their GPUs and load data from our personal Google Drive's 

`error_analysis_richard_and_noah.ipynb` Script reproducing Noah's multimodal model, fixing dataset by removing duplicates, re-running (just test set?) and final results for manuscript.


`multimodal_suicide_detector_with_results.ipynb` final results in paper. 


`semantic_analysis.ipynb` words used in suicidal and nonsuicidal posts