---
layout: single
title:  "Addressing Common Challenges in Deep Learning: Part 1"
date:   2025-06-25 20:00:00 +0100
categories: deep-learning guide
tags: tensorflow
toc: true
---

In this series of posts my goal is to summarize measures on common challenges when
implementing deep feed forward neural networks. These suggestions are not
presented rigorously but rather as hint towards a direction of further actions.
For each challenge i will summarize the underlying idea then present a way to detect this problem and show how to countermeasure.
Most information is taken from [(Goodfellow et al. Deep Learning. MIT Press. 2016)](https://www.deeplearningbook.org/contents/optimization.html),
but supplemented by code implementation.

# Ill-Conditioning of the Hessian

## Basics
The condition number is a measure of sensitivity of a function to changes to its input.
Given a function $$ f : \mathbb{R}^m \rightarrow \mathbb{R}^n $$
and a point $$ x \in \mathbb{R}^m $$ which forms the

**(1)** Absolute Condition Number [^1]

$$
    \begin{split}
        \hat{k}(x) = \underset{\delta \rightarrow 0^+}{\lim} \underset{||h|| < \delta}{\text{sup}}
        \frac{|| f(\boldsymbol{x}+\boldsymbol{h}) - f(\boldsymbol{x}) ||}{ || \boldsymbol{h} ||}
    \end{split}
$$

As in the limit of the change in output over the input.
Given a matrix $$ A $$, the condition number is given by $$ k(a) = ||a|| || A^{-1} || $$, which represents an upper bound.
This can be simplified to the

**(2)** Condition Number of Matrix [^1]

$$ 
 k(A) = \frac{\sigma_{\text{max}}}{\sigma_{\text{min}}}
$$

($$ \sigma $$ represents a singular value. Recall SVD! [^3])

When $$ A $$ is diagonaziable and symmetric (Hermitian) then the condition number is also given as:

$$
    k(A) = \frac{|\lambda_{\text{max}}|}{|\lambda_{\text{min}}|}
$$

Ill conditioning is a very general problem, but it is prominent in gradient based optimization.
Let us consider an SGD update step:

$$
    \begin{split}
        w \leftarrow w - \epsilon \nabla_{w} f(x)
    \end{split}
$$

It is visible that only the first partial derivative dictates the direction of the step.
To analyze this further we apply a Taylor approximation arount the current point $$ \boldsymbol{x}_0 $$ and for small $$ \epsilon $$:

**(3)**
$$
    f(x_0 - \epsilon \boldsymbol{g} ) \approx f(x_0) - \frac{1}{2} \boldsymbol{g}^T \boldsymbol{H} \boldsymbol{g} - \epsilon \boldsymbol{g}^T \boldsymbol{g}
$$

| ![Effect of Curvature](/assets/images/2025-06-25-addressing-common-challenges-dl/1.png) | 
|:--:| 
| *Effects of Curvature:* The dashed line shows sthe expected value based on the gradient. Picture from the book. [^2] |


| ![Effect of Curvature 2](/assets/images/2025-06-25-addressing-common-challenges-dl/2.png) | 
|:--:| 
| *Effect of Ill-Conditioning:* When SGD fails to use the curvature information it can waste time following the steepest feature. This figure represents a long stretched canyon. Picture from the book. [^2] |

The problem arises when the last part (curvature) of equation **(3)** becomes bigger than $$ \epsilon \boldsymbol{g}^T \boldsymbol{g} $$, as in the gradient information.


## How Detect this Issue
We expect as we get to a minimum, that the gradients get small.
Thus we can monitor the $$L^2$$ norm of the gradient and optionally of $$\boldsymbol{g}^T \boldsymbol{H} \boldsymbol{g}$$.
When we observe a relatively big gradient followed by an order of magnitude bigger curvature term, shrinkage of the step size becomes important in order to compensate for the curvature.
Often leading to slow learning.
It advisable to further analyze the problem before simply blaming reaching a stationary point.

Given an already existing Trainer class for a Tensorflow model
we can incorporate the norm calculation by inserting it in a training step.
{% highlight python %}
    class Trainer:
    # (...)
        @tf.function
        def train_step(self, x_batch_train, y_batch_train):
            with tf.GradientTape() as tape:
                pred = self.model(x_batch_train, training=True)
                loss_value = self.loss_fn(y_batch_train, pred)

            grads = tape.gradient(loss_value, self.model.trainable_weights)

            # Monitor the Gradient L2 Norm
            # Flatten the gradients with reshape then concatenate into single vector
            grad_vector = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            # Calculate the Euclidian norm
            grad_norm = tf.norm(grad_vector).numpy()

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            self.train_loss_metric(loss_value)
            self.train_acc_metric.update_state(y_batch_train, pred)

            # Append the Norm Metric of the Step
            self.grad_norm_metric.append(grad_norm)
{% endhighlight %}
The monitoring of the curvature is not shown here, but would be straight forward by using the
supplied functionality of Tensorflow. [^4] [^5]

## Counter Measures
There is definetly not a "one-to-rule-them-all" solution.
Monitoring the gradient norm is a good first step.
This must then be followed by further analyzing why exactly the gradient does
not diminish to 0.

Some simple suggestions that may help solving this issue are the following:
- Use Algorithms with Adaptive Learning Rates
    - AdaGrad
    - RMSProp
    - Adam
    - Others
- Use Schedulers
    - ReduceLROnPlateau
    - Cosine Annealing
    - Others
- Stochastic Gradient Descent (Mini-Batched)

# Next Parts
- Stationary Points
- Flat Regions
- Cliffs
- Long-Term Dependencies
- (...)

# Sources
[^1]: [Applied & Computational Mathematics Emphasis, Conditioning and Stability](https://acme.byu.edu/00000179-aa18-d402-af7f-abf806b20000/conditioning2020-pdf#:~:text=A%20function%20with%20a%20large,produce%20large%20changes%20in%20output.)
[^2]: [Goodfellow et al. Deep Learning. MIT Press. 2016](https://www.deeplearningbook.org/contents/optimization.html)
[^3]: [Peter, Mills. Singular Value Decomposition (SVD) Tutorial: Applications, Examples, Exercises. Medium. 2017](https://medium.com/cube-dev/singular-value-decomposition-tutorial-52c695315254)
[^4]: [Tensorflow, Training Loop](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
[^5]: [Tensorflow, Example Hessian](https://www.tensorflow.org/guide/advanced_autodiff#hessian)