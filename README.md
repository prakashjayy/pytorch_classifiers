### Plant Seedlings classification

Classes present:
-----------------
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet

## Why TARS?
My Favourite character from Interstellar. Its good to use names like murph, cooper, lander etc . 

## TO_DO
1) Selecting the best model (best_model.py)
2) inference.py (Load model, read_image, preprocess, predict)
3) Combining all predictions using different techinques (ensemble.py)
    - Use simple avg
    - Using validation_accuracy as weights and taking weighted average
    - Training a classifier on top of it (NN, randomforest, XGB, LinearRegression)    
### This is an experimental setup to build code base for pytorch. Its main is to experiment faster using transfer learning on all available pre-trained models.

Folder structure:
-----------------
- data
    - train dataset
    - test dataset
    - training_data
        - train
        - valid
    - submission_file

- models
    - All models built
    - Readme.md (Contains the description of each model)
- tars
    - __init__.py
    - tars_models.py
    - tars_data_loaders.py
    - tars_training.py
    - utils.py (Any utility functions keep here)
- submissions
    - once training is completed all the files will be saved here. Model desciption will help
- tar1.py (which contains the training code)
- tar2.py (which contains the training code and extend still)
