# Fashion-MNIST Classification with a Custom Neural Network

Submission:  Atri AI Submission – Fashion-MNIST Assignment 

## Overview

This project builds a fully connected neural network for **Fashion-MNIST image classification** and implements the full training pipeline from scratch, including forward propagation, backpropagation, multiple gradient-based optimizers, and Weights & Biases sweeps for hyperparameter tuning. The assignment also includes a comparison between **cross-entropy loss** and **squared error loss**, plus a small transfer experiment on **MNIST** to test whether the best Fashion-MNIST settings generalize well to a similar dataset.

## What this project contains

* Plotting one sample image from each of the 10 Fashion-MNIST classes.
* A flexible feedforward neural network where the number of hidden layers and hidden units can be changed easily.
* Backpropagation support for **SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam**.
* W&B sweep-based hyperparameter search with a validation split.
* Comparison of **cross-entropy loss** vs **squared error loss**.
* Confusion matrix and evaluation of the best model.
* Transfer of the best configurations to MNIST for a quick generalization test.

## Key ideas used in the implementation

The search strategy narrowed the hyperparameter space by keeping the settings that performed well during preliminary experiments. The report highlights that **ReLU** was preferred over Sigmoid and Tanh, **learning rate 0.001** was the most stable, **hidden layer size 128** gave a good balance of capacity and efficiency, and **early stopping** was used to avoid wasting compute on weak runs.

The best-performing Fashion-MNIST model reported in the submission used **4 hidden layers, 128 hidden units, ReLU, Nadam, learning rate 0.001, batch size 16, Xavier initialization, and 10 epochs**, reaching **87.55% validation accuracy** and **85.14% test accuracy**. 

For the MNIST follow-up experiment, the top three configurations were based on **Adam, Nadam, and Nesterov**, all using the same strong setup: **4 hidden layers, 128 hidden units, batch size 16, ReLU, Xavier initialization, and learning rate 0.001**. They achieved validation accuracies of **96.61%**, **96.63%**, and **96.64%**, with test accuracies of **96.54%**, **96.51%**, and **96.53%** respectively.

The loss comparison section concludes that **cross-entropy and squared error loss produced very similar final accuracies**, although cross-entropy converged a bit faster and was slightly more stable across hyperparameter settings.

## Requirements

Install the Python packages used in the notebook:

```bash
pip install numpy matplotlib tensorflow scikit-learn wandb seaborn pandas
```

## How to run

1. Clone or download the project.
2. Open the notebook `fashion_mnist_assignment.ipynb`.
3. Install the dependencies listed above.
4. Log in to Weights & Biases:

   ```bash
   wandb login
   ```
5. Run the notebook cells in order:

   * load and visualize Fashion-MNIST,
   * define activations and weight initialization,
   * implement forward/backpropagation,
   * run the optimizer experiments,
   * launch the W&B sweep,
   * evaluate the best model,
   * compare loss functions,
   * run the MNIST transfer experiment.

## Project structure

* **Data loading and visualization**: loads Fashion-MNIST and displays one example per class.
* **Model code**: defines the feedforward network, activations, initialization, and backpropagation.
* **Optimizers**: includes SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam.
* **Experiment tracking**: uses Weights & Biases for sweeps and run logging.
* **Evaluation**: reports accuracy, loss, and confusion matrix.
* **Extra experiment**: reuses the learned hyperparameters on MNIST.

## Results summary

* Best Fashion-MNIST validation accuracy: **87.55%**
* Best Fashion-MNIST test accuracy: **85.14%**
* MNIST validation accuracy with top configs: about **96.6%**
* MNIST test accuracy with top configs: about **96.5%**


