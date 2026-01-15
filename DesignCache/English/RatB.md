# RatB - Ratio Based

## Definition

The AdaLR or 'Adaptive Learning Rate' concept is a subset of learning rate schedulers with adaptive properties.
AdaLR, as a scheduling concept, has a striking advantage: its ability to adjust the learning rate based on a specified benchmark.

**RatBLR** (Ratio-Based Learning Rate) is a learning rate scheduler based on the loss ratio. This concept is based on the basic idea of ​​an individual who stated that a learning rate scheduler should have the ability to adapt in real time based on the loss rate.

## Theoretical Support

My personal opinion from the results of my observations and observations states that the greater the loss, the more chaotic the model will be during the training process, and the learning rate as a regulator of the model's 'movement' steps in capturing gradient signals has the potential to restore the situation by setting the loss in the right direction, namely a direction that can reduce loss. In my observations, when loss increases, a larger lr is consistently able to reduce loss. Theoretically this could be because a larger learning rate is able to pick up more gradient signals which is expected to be able to reduce loss.

## Implementation

RatBLR works by relying on the learning rate using the loss ratio in the model training process. RatBLR can be roughly denoted as:

$\text{lr_rate}_t = \text{lr_rate}_{t-1} \cdot \left( \frac{\text{loss}_{t-1}}{\text{loss}_{t-2}} \right)$

The new learning rate is derived from the previous index learning rate multiplied by the loss ratio by dividing $\text{loss}_{t-1}$ by $\text{loss}_{t-2}$. This division produces a mathematical property: if $\text{loss}_{t-1} > \text{loss}_{t-2}$, then
the learning rate multiplied by the ratio will increase according to the result of the loss division. Conversely, if $\text{loss}_{t-1} < \text{loss}_{t-2}$, then multiplying by the loss ratio will reduce the learning rate, assume the value of the general loss function in classical machine learning is non-negative.
This definition makes RatBLR quite effective in handling datasets with high noise and has the potential to be an effective learning rate scheduler in some cases.

From the explanation above, the RatBLR implementation can be further expanded by adding a 'window' to the loss to obtain a wider and more even ratio.
This extended implementation can be denoted as follows (using basic Python programming language syntax):

$\text{lr_rate}_t = \text{lr_rate}_{t-1} \cdot \frac{\text{mean}(\text{loss}[-window:])}{\text{mean}(\text{loss}[-2window:-window])}$

This implementation allows RatBLR to better evaluate the steps involved in increasing and decreasing the learning rate.
However, this formula is still inefficient because if the loss ratio is only around 1, the increase and decrease in the learning rate will be very small and slow down the training process.
To overcome this problem, the final formula can be strengthened by changing it to:

$$
\text{lr} =
\begin{cases}
\text{lr} \cdot (i \cdot t^{p}), & \text{if } r \leq 1 \\
\text{lr} \cdot r, & \text{if } r > 1
\end{cases}
$$

where

$$
r = \sqrt{ \frac{\text{mean}(\text{loss}[-window:])}{\text{mean}(\text{loss}[-2window:-window])} }
$$

This formula will better protect the ratio from overflow and maintain a contributive learning rate.
A ratio below 1 will cause the learning rate to be reduced using the invscaling formula to create a smooth and regular decrease in the learning rate.
A ratio above 1 will cause the learning rate to be increased according to the ratio. The rooted ratio can be assumed to scale well and safely, reducing the possibility of learning rate overflow during the increase.

The natural sensitivity of RatBLR theoretically matches linear models like the gradient descent family due to its relatively smooth loss and easy explained representation when compared to neural networks (NN) and other types of nonlinear models.

Despite the advantages of the concept outlined above, the nonlinearity of RatBLR can potentially produce unexpected results, such as infinity or NaN (Not a Number). This is because the RatBLR formula lacks a direct mathematical constraint.
Therefore, in practice, RatBLR uses clipping to prevent unexpected values.

The RatBLR concept is designed and specifically designed for classical machine learning, where fluctuations in loss can be explained by its linear nature.

## Deeper Implementation

This deeper implementation aims to expand the use cases of the AdaLR concept described above. This deeper implementation aims to make the AdaLR concept compatible with more sophisticated types of machine learning, such as neural networks (NN).

NNs are known for their highly unstable loss characteristics, and their nonlinearity adds complexity to NNs in terms of loss representation.

Because the RatBLR learning rate scheduler is theoretically unstable when implemented in a NN system, the non-linearity of the loss in the NN can disrupt the loss ratio, causing RatBLR to adjust the learning rate unstably.
To address this issue, the RatBLR formula has been slightly modified to accommodate the characteristics of NNs. The most fundamental change is that the ratio is now scaled using a logarithm instead of a square root and conversion to absolute value to prevent the ratio from changing lr sign when multiplied because negative loss is not impossible and common in NN, which can be denoted as:

$$
r = \abs( \log({ \frac{\left(\text{mean}(\text{loss}[-window:])\right)}{\text{mean}(\text{loss}[-2window:-window])} }) )
$$

This formula allows for a more stable and well-organized ratio, potentially reducing the algorithm's sensitivity.

The ratio scale affected by the logarithm is sufficient to increase stability. However, once again, in NN systems, the loss is highly fluctuating. To increase the stability of the algorithm, this formula can be enhanced by a "patience" system.
The patience system works by restraining the algorithm's behavior in adjusting the learning rate even after it reaches the threshold ratio until a predetermined patience limit is reached before the algorithm can finally adjust the learning rate. With the addition of the patience mechanism, the algorithm notation can be defined as:

$$
\text{lr} =
\begin{cases}
\text{lr} \cdot (i \cdot t^{p}), & \text{if } r \leq 1 \text{ and } \text{wait} \geq \text{patience} \\
\text{lr} \cdot r, & \text{if } r > 1 \text{ and } \text{wait} \geq \text{patience}
\end{cases}
$$

where

$$
r = \abs( \log({ \frac{\left(\text{mean}(\text{loss}[-window:])\right)}{\text{mean}(\text{loss}[-2window:-window])} }) )
$$

$\text{wait}$ It functions as a "patience count" for the algorithm when the ratio reaches its threshold.
With the addition of a logarithmic mechanism and stability, the above algorithm can theoretically be used and optimized for developing small, medium, and even large neural networks (NNs).
The final definition of the algorithm can be called RobustRat, or Robust Ratio, which describes the algorithm's characteristics.

## Conclusion
The learning rate scheduler is crucial in machine learning because it directly influences how parameters such as weights or coefficients are updated. The **Ratio-Based (RatB)** concept, which includes the **Ratio-Based Learning Rate (RatBLR)** and the **Robust Ratio (RobustRat)**, as a more stable form for neural networks (NNs), provides a learning rate scheduler option capable of creating an adaptive yet still contributive effect during model training.