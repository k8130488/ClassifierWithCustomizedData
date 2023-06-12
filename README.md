# ClassifierWithCustomizedData
 Practice training Pytorch CNN using your own data set.
## Environment
<ul>
<li>python >= 3.6.9</li>
<li>CUDA >= 10.2</li>
<li>cuDNN >= 7.5.6</li>
</ul>

## Requirements
`pip install -r requirements.txt`

## Prepare Dataset
The Dataset must be prepared by yourself and placed in the following path.
```buildoutcfg
-dataset/
    |-dogs_vs_cats/
        |-cats/
            |-train
            └-test
        |-dogs/
            |-train
            └-test
    |-your dataset/
        |-class A/
            |-train
            |-valid
            └-test
        |-class B/
            |-train
            |-valid
            └-test
        └-...
    └-...
```

## Train
You can train your model with train.py
`python train.py --source dataset --image-set dogs_vs_cats --epochs 100 --model alexnet --early-stop True`

## Test
You can get the performance of the model, like confusion matrix, PR curve, and ROC curve, with test.py
`python test.py --source dataset --image-set dogs_vs_cats --image-size 320 --model alexnet`

## Other Modifications
Here is only a preliminary introduction to how to add the required model or loss function

### Model
You can create and modify the model you need in utils/models.py, you need to change the code block that calls the model in train.py. Only simple examples of ResNet50 and AlexNet are provided here.

### Loss Function
You can create and modify the loss function you need in utils/loss.py, including the loss function provided by torch, and you need to change the code block that calls the model in train.py or run the error point. Only a simple focal loss example is provided here.

