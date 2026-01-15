# Logic Limit That We Missed

## Introduction and Definition

Cognitive function is the ability to manage and align information to make it useful for a specific purpose. This information management and alignment refers to the ability to allocate relevant information to a specific goal and to adapt information from different domains without losing its meaning. In the context of machine learning, cognitive function extension refers to how machines are presented with diverse information across domains and align it for prediction purposes. This is generally not prioritized due to the perceived trade-offs of interpretability and systemic complexity. Some architectures prioritize cognitive depth within a single function, rather than its extension.

Machine learning, or ML, is the moment when a program constructs an internal representation of data, capable of generating new responses that are not explicitly programmed but are still consistent with learned data patterns.

Some key elements of machine learning include the error function (loss function), the derivative of the error function, and the number of steps taken to minimize the error (loss). Deep learning (DL), a branch of ML, is capable of solving more complex patterns and requires a more complex architecture than classical ML. Architecturally, DL enables the recognition of non-linear patterns during the training process. But does DL reflect an expansion of cognitive function?

Architecturally, DL generally only deepens one cognitive function because neural networks are trained with one information paradigm or one type of error (loss). From this, it can be seen that cognitive expansion does not originate from complex neural network architectures; it stems not only from how information is utilized but also from how it is managed and aligned with other information across domains.

**Naïve Specialist Hierarchy**, or **NSH**, is an epistemic orchestration architecture that enables the expansion of cognitive functions, introducing a new paradigm in machine learning, where the priority is cognitive expansion, not just the deepening of a single cognitive function.

## Architecture Structure

**NSH** consists of a collection of models specialized in different information domains, and a single umbrella system encompasses all these models, or specialists, within the system. Each specialist operates as an independent model specialized in a specific information domain, while the umbrella system serves as a structural coordinator rather than a predictive model, providing a unified interface for the heterogeneous specialists.

## Architecture Mechanism

### A. Preprocessing

The user initiates feature data allocation, and the umbrella distributes feature data according to the allocation index to each model. This creates a specialization effect within each model. This specialization effect allows **NSH** to distinguish between optimal and non-optimal specialized models based on user input.

### B. Processing and "Trust"

The umbrella can filter specialists through a trust mechanism that influences the output of each specialist. Each specialist is trusted based on the magnitude and stability of errors, skill scores on validation data during the training process, and calibration during the evaluation process. Specialists with trust > 1 can be called *prior* (priority), trust ~1 is called *major* (majority), and trust < 0 is called *frozen*, frozen specialists do not participate in the prediction process until trust > 0 on the next user input data. The trust of each specialist can change depending on its performance on validation data during the training session or on evaluation data input by the user. Also, trust can decrease as the user calls **NSH** to make predictions.

### C. Predicting

Each prior and major generate predictions from user input data before the umbrella stores the output representation data of all specialists, which is then multiplied by each specialist's trust, until each specialist's predictions are summed to form the final prediction.

### D. Evaluating

Umbrella is able to evaluate the trustworthiness of each specialist using the final prediction data through a function that allows each specialist to predict labels from the given data. Then, performance metrics such as performance loss can be assessed. If there is improvement from previous training, trust can increase based on the performance difference. Conversely, if the evaluation results indicate a decline in a specialist's performance, the model's trust will decrease based on the performance difference.

## Hypothesis

Hypothetically, the **NSH** architecture enables the expansion of information management and alignment capabilities through a collection of specialized models where specialists are structurally isolated from each other, thus theoretically mitigation catastrophic forgetting when new information enters. It is also theoretically capable of maintaining alignment between specialists through trust decay, even with a large number of specialists.

The **NSH** architecture opens up several new possibilities. Theoretically, it enables the implementation of online learning based on an umbrella structure that allows for the addition of specialists without catastrophic forgetting because the addition of new information from new specialists is isolated from the existing information from existing specialists. **NSH** with online learning or **ONSH (Online Naïve Specialists Hierarchy)** is very possible with several additional mechanisms, such as: 1. separation of old specialist categories and new specialists (early-specialists), 2. trust in new specialists who have different functions from old specialists but still in the same basic concept of the formula, the purpose of trust in early-specialists is as a measure of eligibility to be part of the existing specialists, if the early-specialist trust > 1 then it becomes part of the permanent specialist from paying, if trust ~ 1 with a certain threshold then the early-specialist will be frozen until the trust increases in the next user input, and if trust < 0 then the early-specialist will be deleted., 3. The emergence of early-specialists only when all old specialists have trust < 0 or Out Of Domain.

## Conclusion

In closing, the world of machine learning is so vast in the current era of AI technology development. **NSH** introduces a new model management paradigm in machine learning where the expansion of cognitive functions is also prioritized, not only its deepening. Some consider cognitive expansion in ML models an unnecessary systemic burden, but it is important to understand that this expansion is not just an addition of meaningless structural complexity but an implicit interpretation of data meaning through internal representation, management and alignment of information that allows 1 structure to be able to handle various types, domains, and paradigms of information.