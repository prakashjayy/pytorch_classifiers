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

Models        | Train Accuracy_score |    Val Accuracy_score |
------------- | :------------------: | --------------------: |
resnet18      |    0.77553           |    0.75518            |
resnet152     |    0.82778           |    0.81535            |
resnet101     |    0.82333           |    0.80290            |
resnet50      |    0.79943           |    0.78630            |
resnet34      |    0.78655           |    0.74688            |
squeezenet1_0 |    0.91447           |    0.87966            |
squeezenet1_1 |    0.90089           |    0.87344            |
densenet121   |    0.80880           |    0.81120            |
densenet169   |    0.84746           |    0.82987            |
densenet201   |    0.86621           |    0.86514            |
inception_v3  |    0.76101           |    0.74688            |
vgg11         |    0.78209           |    0.78008            |
vgg13         |    0.75960           |    0.72821            |
vgg16         |    0.77038           |    0.71576            |
vgg19         |    0.71204           |    0.64522            |
vgg11_bn      |    0.76522           |    0.74481            |
vgg13_bn      |    0.76241           |    0.76348            |
vgg16_bn      |    0.76265           |    0.75726            |
vgg19_bn      |    0.75773           |    0.73858            |
alexnet       |    0.83153           |    0.76348            |


## Finetuning the entire network freeze=False
Models        | Train Accuracy_score |    Val Accuracy_score |
------------- | :------------------: | ----------------------:
resnet18      |    0.98477           |    0.96058            |
resnet152     |    0.99273           |    0.97717            |
resnet101     |    0.99367           |    0.97717            |
resnet50      |    0.99133           |    0.97510            |
resnet34      |    0.98969           |    0.97095            |
squeezenet1_0 |    0.96274           |    0.94190            |
squeezenet1_1 |    0.96485           |    0.92738            |
densenet121   |    0.99086           |    0.96887            |
densenet169   |    0.99507           |    0.97510            |
densenet201   |    0.99390           |    0.97717            |
inception_v3  |    0.98898           |    0.97302            |
vgg11         |    0.98031           |    0.95020            |
vgg13         |    0.98078           |    0.95643            |
vgg16         |    0.98266           |    0.95435            |
vgg19         |    0.98430           |    0.95643            |
vgg11_bn      |    0.98570           |    0.96265            |
vgg13_bn      |    0.98687           |    0.97095            |
vgg16_bn      |    0.99179           |    0.96680            |
vgg19_bn      |    0.99297           |    0.96680            |
alexnet       |    0.95970           |    0.92946            |


**Added resnext101_32x4d, resnext101_64x4d, inceptionv4, inceptionresnetv2, nasnetalarge, bninception and vggm**

Submissions:
- densenet201 LB - to 97.22
- ensemble1 - mode of all best performing models LB - 97.32

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
