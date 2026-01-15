# Awareness Like

## Introduction

Machine learning (ML) is the moment when a program is able to produce results that it previously didn't see during the training process.
Some key elements in machine learning include the error function (loss function), its derivatives, and the number of steps taken to minimize the error (loss).
The size of this "step" can be determined by the specified learning rate. The learning rate in machine learning plays a crucial role in minimizing model error.
The learning rate in machine learning models, particularly linear gradient models, works by multiplying the gradient of the error function before influencing parameters such as weights or coefficients. This can generally be denoted as:

W_t = W_t-1 - lr * grad_w

The example above represents the most common notation in machine learning when a model is in the training process.

The learning rate itself varies depending on the scheduler used. Generally, learning rate schedulers have the following properties:

- Constant
- Decay
- Adaptive

## Deepening

The learning rate properties during training are usually chosen according to user needs, considering the alignment of model characteristics with the learning rate properties, the assumed stability of the error function (loss) during training, and other factors. The learning rate properties, each with its own characteristics, can be described as follows:

### Constant
The learning rate does not change during the training process. The scheduler is generally defined as 'Constant,' which is generally used when interpretability and clarity of model performance are prioritized.

### Decay
The learning rate decreases as the training process progresses.
An example of this scheduler is 'Invscaling,' which exponentially reduces the learning rate and is generally used by default by models with convex error functions.

### Adaptive
The learning rate changes according to the movement of a parameter. An example of this is the "LROnPlateau" scheduler, which will decrease the learning rate when the error reduction progress stagnates. It is generally used during long training runs with many epochs, and performance stagnation is a major issue.

Some schedulers also have a "programmed" LR formula during the training process, such as a single cycle.

## Opinion

The LR properties of the scheduler are intended to address specific problems or general defaults, but I feel some popular schedulers are too fixated on LR decrement and the constraints of mathematical formulas. I feel there should be a scheduler that can "understand" the training situation and adapt accordingly.

> CoDe: 1. **RatB (Cached)**, 2. **ConB**
> Paper: -