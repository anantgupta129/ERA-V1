# Neural Network BackPropagation on Excel

[Link to Excel Sheet](./Mathematics%20behind%20Backpropogation.xlsx)

Backpropagation is a commonly used method for training a neural network. The goal of backpropagation is to optimize the weights of the network so the neural network can learn how to correctly map the input and output attributes. The excel sheet helps to understand backpropagation correctly. Here, we’re going to use a neural network with two inputs, two hidden neurons, two output neurons and we are ignoring the bias.

![w10 15](https://user-images.githubusercontent.com/84603388/120035598-7fc27600-c01c-11eb-8afd-ef9f49a8bcdb.png)

Here are the initial weights, for us to work with:

```
w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3
w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55
```

A single training set is used in the excel sheet: given inputs 0.05 and 0.10, we want the neural network to output 0.01 and 0.99.

## Forward Propagation

At first, the above inputs are passed through the network by multiplication of inputs and weights and the h1 and the h2 is are calculated

```
  h1 =w1*i1+w2+i2
  h2 =w3*i1+w4*i2
```

The hidden layer neurons output which is h1 and h2 are passed to activation function using sigmoid activation, output represented by a_h1 and a_h2, this helps in adding nonlinearity to the network.

```
  a_h1 = σ(h1) = 1/(1+exp(-h1))
  a_h2 = σ(h2) = 1/(1+exp(-h2))
```

This process is repeated for the output layer neurons, using the output from the hidden layer activated neurons as inputs.

```
  o1 = w5 * a_h1 + w6 * a_h2
  o2 = w7 * a_h1 + w8 * a_h2

  a_o1 = σ(o1) = 1/(1+exp(-o1))
  a_o2 = σ(o2) = 1/(1+exp(-o2))
```

Next, The error for each output neurons (a_o1 and a_o2) is calculated using the squared error function and sum them up to get the total error (E_total)

## Calculating the Error (Loss)

```
E1 = ½ * ( t1 - a_o1)²
E2 = ½ * ( t2 - a_o2)²
E_Total = E1 + E2
```

Note:  1/2 is included so that exponent is cancelled when we differentiate the error term.

## Back Propagation

During back propagation, the network learn and get better by updating the weights such that the total error is minimized

Partial derivative of E_total with respect to w5, w6, w7, and w8

```
δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2
```

The same goes for w2, w3, and w4

```
δE_total/δw1 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2
```

Once the gradient is  calculated, the weights are updated using the learning rate with this equation

```
    w = w - learning_rate * δE_total/δw
```

The process is repeated until we get the minimum loss

## Error Graph for different Learning rates

Below is the error graph when we change the learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0

![Screenshot (270)](https://user-images.githubusercontent.com/84603388/120029766-27877600-c014-11eb-9420-23c1748c5579.png)

### Conclusion: The small learning rate takes a lot of time to find the optimum minimum value and the large learning rate takes a lot of jumps and gets stuck to find the minimum value. So, we should choose a moderate value for learning rate
