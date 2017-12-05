""" This repo will ensemble all the models in different ways

We will be using the following ensembling strategies:
- Select the model outputs you want to use.
- Take Voting and choose the class with most number of Voting
"""

import pandas as pd


models_used = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201",
"squeezenet1_0", "squeezenet1_1", "inception_v3", "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "alexnet"]
models_loc = ["submission/"+i+"_False_freeze.csv" for i in models_used]

models_combined=pd.DataFrame()
cols = ["file", "species"]
for i in models_loc:
    f = pd.read_csv(i)
    f["file"] = f["img_loc"].apply(lambda x:x.rsplit("/")[-1])
    f = f[["file", "label"]]
    models_combined["file"] = f["file"].values
    models_combined[i.rsplit("/")[-1].rsplit(".")[0]] = f["label"].values

print(models_combined.shape)
print(models_combined.head())
models_combined.to_csv("submissions/ensemble1_intermediate.csv", index=False)
cols = list(models_combined.columns)
cols.pop(0)
models_combined["species"] = models_combined[cols].mode(axis=1)
models_combined[["file", "species"]].to_csv("submissions/ensemble.csv", index=False)
