## This is an experimental setup to build code base for pytorch. Its main is to experiment faster using transfer learning on all available pre-trained models.


### Why TARS?
My Favourite character from Interstellar. Its good to use names like murph, cooper, lander etc .



### Dataset: Plant Seedlings classification

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

#### Run the following commands
To train the models:
```
make -f Makefile.train
```

To test the models:
```
make -f Makefile.predict
```


## Results
total number of train images: 4268
total number of val images: 482
total number of test images: 794

## Freezed all the layers freeze=True
resnet18      | Accuracy_score train : 0.77553 | Accuracy_score val : 0.75518  
resnet152     | Accuracy_score train : 0.82778 | Accuracy_score val : 0.81535  
resnet101     | Accuracy_score train : 0.82333 | Accuracy_score val : 0.80290  
resnet50      | Accuracy_score train : 0.79943 | Accuracy_score val : 0.78630  
resnet34      | Accuracy_score train : 0.78655 | Accuracy_score val : 0.74688  
squeezenet1_0 | Accuracy_score train : 0.91447 | Accuracy_score val : 0.87966  
squeezenet1_1 | Accuracy_score train : 0.90089 | Accuracy_score val : 0.87344  
densenet121   | Accuracy_score train : 0.80880 | Accuracy_score val : 0.81120  
densenet169   | Accuracy_score train : 0.84746 | Accuracy_score val : 0.82987  
densenet201   | Accuracy_score train : 0.86621 | Accuracy_score val : 0.86514  
inception_v3  | Accuracy_score train : 0.76101 | Accuracy_score val : 0.74688  
vgg11         | Accuracy_score train : 0.78209 | Accuracy_score val : 0.78008  
vgg13         | Accuracy_score train : 0.75960 | Accuracy_score val : 0.72821  
vgg16         | Accuracy_score train : 0.77038 | Accuracy_score val : 0.71576  
vgg19         | Accuracy_score train : 0.71204 | Accuracy_score val : 0.64522  
vgg11_bn      | Accuracy_score train : 0.76522 | Accuracy_score val : 0.74481  
vgg13_bn      | Accuracy_score train : 0.76241 | Accuracy_score val : 0.76348  
vgg16_bn      | Accuracy_score train : 0.76265 | Accuracy_score val : 0.75726  
vgg19_bn      | Accuracy_score train : 0.75773 | Accuracy_score val : 0.73858  
alexnet       | Accuracy_score train : 0.83153 | Accuracy_score val : 0.76348  


## Finetuning the entire network freeze=False
resnet18      | Accuracy_score train : 0.98477 | Accuracy_score val : 0.96058  
resnet152     | Accuracy_score train : 0.99273 | Accuracy_score val : 0.97717  
resnet101     | Accuracy_score train : 0.99367 | Accuracy_score val : 0.97717
resnet50      | Accuracy_score train : 0.99133 | Accuracy_score val : 0.97510  
resnet34      | Accuracy_score train : 0.98969 | Accuracy_score val : 0.97095  
squeezenet1_0 | Accuracy_score train : 0.96274 | Accuracy_score val : 0.94190  
squeezenet1_1 | Accuracy_score train : 0.96485 | Accuracy_score val : 0.92738  
densenet121   | Accuracy_score train : 0.99086 | Accuracy_score val : 0.96887  
densenet169   | Accuracy_score train : 0.99507 | Accuracy_score val : 0.97510  
densenet201   | Accuracy_score train : 0.99390 | Accuracy_score val : 0.97717  
inception_v3  | Accuracy_score train : 0.98898 | Accuracy_score val : 0.97302  
vgg11         | Accuracy_score train : 0.98031 | Accuracy_score val : 0.95020  
vgg13         | Accuracy_score train : 0.98078 | Accuracy_score val : 0.95643  
vgg16         | Accuracy_score train : 0.98266 | Accuracy_score val : 0.95435  
vgg19         | Accuracy_score train : 0.98430 | Accuracy_score val : 0.95643  
vgg11_bn      | Accuracy_score train : 0.98570 | Accuracy_score val : 0.96265  
vgg13_bn      | Accuracy_score train : 0.98687 | Accuracy_score val : 0.97095  
vgg16_bn      | Accuracy_score train : 0.99179 | Accuracy_score val : 0.96680  
vgg19_bn      | Accuracy_score train : 0.99297 | Accuracy_score val : 0.96680  
alexnet       | Accuracy_score train : 0.95970 | Accuracy_score val : 0.92946  

## TO_DO
3) Combining all predictions using different techinques (ensemble.py)
    - Use simple avg
    - Using validation_accuracy as weights and taking weighted average
    - Training a classifier on top of it (NN, randomforest, XGB, LinearRegression)    

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
- tars_train.py (which contains the training code)
- tars_predict.py (which contains the training code and extend still)
