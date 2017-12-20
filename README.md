## This is an experimental setup to build code base for pytorch. Its main is to experiment faster using transfer learning on all available pre-trained models.

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


# Results with Full-Agumentation strategy:

## Trained the networks in three methods:
   - Full Finetuning
   - Freeze first few layers

## Case-1 - Finetuning entire network
Models           | Train Accuracy_score |    Val Accuracy_score |
-------------    | :------------------: | --------------------: |
resnet18         |    0.92783           |    0.93153            |
resnet34         |    0.9522            |    0.94190            |
resnet50         |    0.95665           |    0.94398            |
resnet101        |    0.96696           |    0.96265            |
resnet152        |    0.96555           |    0.95643            |
squeezenet1_0    |    0.94329           |    0.92738            |
squeezenet1_1    |    0.93955           |    0.93153            |
densenet121      |    0.95243           |    0.9336             |
densenet169      |    0.96626           |    0.93983            |
densenet201      |    0.96063           |    0.95020            |
inception_v3     |    0.94212           |    0.93568            |
vgg11            |    0.93814           |    0.93153            |
vgg13            |    0.94493           |    0.94190            |
vgg16            |    0.95665           |    0.93568            |
vgg19            |    0.95009           |    0.93775            |
vgg11_bn         |    0.94142           |    0.93775            |
vgg13_bn         |    0.94423           |    0.92738            |
vgg16_bn         |    0.94634           |    0.94190            |
vgg19_bn         |    0.94915           |    0.93360            |
alexnet          |    0.91260           |    0.90456            |
resnext101_64x4d |    0.98055           |    0.96887            |
resnext101_32x4d |    0.98172           |    0.96887            |
nasnetalarge     |    0.96907           |    0.96265            |
inceptionresnetv2|    0.96134           |    0.95435            |
inceptionv4      |    0.96930           |    0.96473            |


## Case-2 - Freezed first few layers(look at code)
Models           | Train Accuracy_score |    Val Accuracy_score |
-------------    | :------------------: | ----------------------:
resnet18         |    0.9196            |    0.91493            |
resnet34         |    0.94845           |    0.93983            |
resnet50         |    0.9564            |    0.93983            |
resnet101        |    0.96790           |    0.96265            |
resnet152        |    0.96508           |    0.95643            |
squeezenet1_0    |    0.94048           |    0.92738            |
squeezenet1_1    |    0.93088           |    0.92116            |
densenet121      |    0.95173           |    0.95228            |
densenet169      |    0.96087           |    0.94813            |
densenet201      |    0.95384           |    0.95020            |
inception_v3     |    0.94025           |    0.93775            |
vgg11            |    0.93697           |    0.92946            |
vgg13            |    0.93533           |    0.92323            |
vgg16            |    0.94821           |    0.93983            |
vgg19            |    0.95243           |    0.94190            |
vgg11_bn         |    0.93416           |    0.92738            |
vgg13_bn         |    0.93322           |    0.92323            |
vgg16_bn         |    0.94728           |    0.93775            |
vgg19_bn         |    0.94798           |    0.94190            |
alexnet          |    0.89784           |    0.88589            |
resnext101_64x4d |    0.98617           |    0.96887            |
resnext101_32x4d |    0.98195           |    0.96473            |
nasnetalarge     |    0.95970           |    0.96265            |
inceptionresnetv2|    0.95103           |    0.94813            |
inceptionv4      |    0.96251           |    0.93775            |


# Results with Basic Agumentation

## Case-1 (Freezed all layers except last one)
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


## Case-2 Finetuning the entire network
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


## Submissions:
- densenet201 LB - to 97.22
- ensemble1 - mode of all best performing models LB - 97.32

## TO_DO
1) Adding mixup strategy to all the networks
2) Ensembling model outputs
3) Model stacking
4) Extracting bottleneck features and using
        - ML to train the model
        - Visualization using T-sne
5) Solve issue with bninception(Model is not training)
6) Train Vggm network 
