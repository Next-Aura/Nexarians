# NSH - Naive Specialist Hierarchy

## Definition

NSH, or Naive Specialist Hierarchy, is a concept that introduces a machine learning architecture that enables the expansion of cognitive functions.

## Theoretical Support

Cognitive function expansion is one way to generalize. A neural network (NN) mathematically sharpens only one cognitive function, leading to the assumption that the expansion of cognitive capabilities is not based on a large/deep NN but rather on training techniques and the harmonization of information across paradigms within the architecture.

## Architectural Structure

NSH consists of a collection of models, each receiving different portions of information, and a layer that encompasses all models in the system, which can be referred to as an "umbrella."

## Training

### Pre-processing

The umbrella regulates what information will be provided to all models (feature data), the feature data is cut based on the richness of the information, and the variation of the feature information. This creates a specialization effect for each model. This specialization effect is what allows NSH to distinguish between optimal and non-optimal models for user input.

### Process and Trust

Umbrella can filter models through a *trust* mechanism that can influence the output of each model. Each model has a trust based on the magnitude and stability of its loss, its skill score on validation data during the training process, and the model's confidence in its output. Models with trust >1 are referred to as prior (priority), trust ~1 is called major (majority), and <0 is called frozen. Frozen models do not participate in the prediction process until trust >0 on subsequent user input. The trust of each model can fluctuate depending on its performance on user input data. Furthermore, trust can decrease as the user invokes NSH to make predictions; high trust decays more rapidly than models with low trust.

## Prediction

Umbrella produces the final prediction by storing data representing the output of all models, which is then multiplied by each model's trust before finally summing them into the final prediction.

## Evaluation

Umbrella is able to evaluate itself using its prediction data, with a function that allows each model to predict labels from the given data. Then, performance metrics such as loss and performance are assessed. If there is improvement from previous training, trust can increase based on the performance difference. Conversely, if the evaluation results indicate a decline in a model, trust will decrease based on the performance difference.