# MNIST less then 20k

[Notebook](./S6_net1.ipynb)
- There are total **16,034** parameters in networks.
- Dropout of **0.01** is used every layer (except after fully connected layer).
- Achieved validation accuracy of **99.4%** in 17 epoch with max **99.43%**

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 26, 26]           1,152
              ReLU-6           [-1, 16, 26, 26]               0
       BatchNorm2d-7           [-1, 16, 26, 26]              32
         Dropout2d-8           [-1, 16, 26, 26]               0
            Conv2d-9           [-1, 16, 24, 24]           2,304
             ReLU-10           [-1, 16, 24, 24]               0
      BatchNorm2d-11           [-1, 16, 24, 24]              32
        Dropout2d-12           [-1, 16, 24, 24]               0
        MaxPool2d-13           [-1, 16, 12, 12]               0
           Conv2d-14            [-1, 8, 12, 12]             128
             ReLU-15            [-1, 8, 12, 12]               0
        Dropout2d-16            [-1, 8, 12, 12]               0
           Conv2d-17           [-1, 16, 10, 10]           1,152
             ReLU-18           [-1, 16, 10, 10]               0
      BatchNorm2d-19           [-1, 16, 10, 10]              32
        Dropout2d-20           [-1, 16, 10, 10]               0
           Conv2d-21             [-1, 32, 8, 8]           4,608
             ReLU-22             [-1, 32, 8, 8]               0
      BatchNorm2d-23             [-1, 32, 8, 8]              64
        Dropout2d-24             [-1, 32, 8, 8]               0
           Conv2d-25              [-1, 8, 8, 8]             256
             ReLU-26              [-1, 8, 8, 8]               0
        Dropout2d-27              [-1, 8, 8, 8]               0
           Conv2d-28             [-1, 16, 6, 6]           1,152
             ReLU-29             [-1, 16, 6, 6]               0
      BatchNorm2d-30             [-1, 16, 6, 6]              32
        Dropout2d-31             [-1, 16, 6, 6]               0
           Conv2d-32             [-1, 32, 4, 4]           4,608
             ReLU-33             [-1, 32, 4, 4]               0
      BatchNorm2d-34             [-1, 32, 4, 4]              64
        Dropout2d-35             [-1, 32, 4, 4]               0
        AvgPool2d-36             [-1, 32, 1, 1]               0
           Linear-37                   [-1, 10]             330
================================================================
Total params: 16,034
Trainable params: 16,034
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.00
Params size (MB): 0.06
Estimated Total Size (MB): 1.07
----------------------------------------------------------------
```

## Learning Curve 

![](./images/learning_curve.png)

![](./images/output.png)