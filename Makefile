python:
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet18" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet18" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet34" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet34" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet50" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet50" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet101" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet101" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet152" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "resnet152" -f -ep 100 -b 8
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet121" -ep 100 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet121" -f -ep 100 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet169" -ep 100 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet169" -f -ep 100 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet201" -ep 100 -b 8
	python model1.py -idl "data/training_data" -sl "models/" -mo "densenet201" -f -ep 100 -b 8
	python model1.py -idl "data/training_data" -sl "models/" -mo "squeezenet1_0" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "squeezenet1_0" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "squeezenet1_1" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "squeezenet1_1" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "inception_v3" -ep 100 -is 299 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "inception_v3" -f -ep 100 -is 299 -b 16
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg11" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg11" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg13" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg13" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg16" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg16" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg19" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg19" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg11_bn" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg11_bn" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg13_bn" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg13_bn" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg16_bn" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg16_bn" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg19_bn" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "vgg19_bn" -f -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "alexnet" -ep 100
	python model1.py -idl "data/training_data" -sl "models/" -mo "alexnet" -f -ep 100
