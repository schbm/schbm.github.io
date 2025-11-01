---
layout: single
title:  "Standardizing the Temporal Axis"
date:   2025-10-31 17:00:00 +0100
categories: ds ml guide
toc: false
---

As most machine learning practitioners know, a large part of the work in traditional ML doesn’t lie in the modeling itself, but in the preprocessing and feature engineering that come before it. One particularly tricky situation arises when dealing with wide-format experimental or process data where each test forms a row and each stage becomes its own column.

I recently ran into exactly this challenge while developing a regression model. The test produced measurements at fixed time intervals but stopped at different times for each run, resulting in variable data lengths.

The dataset looked similar to the table below:

| Age | Type | Stage1 | Stage2 | (...) | StageX |
|-----|------|--------|--------|-------|--------|
| 20  |  1   | 1      | 2      | ...   | NaN    |
| 25  |  2   | 1      | NaN    | ...   | NaN    |
| 30  |  1   | 1      | 2      | ...   | 3      |

So there I was, asking the question:
How do you squash this into few meaningful variables without throwing away too much information?
As the first step we can apply a nice thing called **domain-knowledge**.

In our particular case I knew the individual tests could be modelled using exponential functions.
But this does not solve the problem of variable length and the possibility of wildly different curve morphology.


{% include curves.html %}
* Before normalization curves differ greatly. The colors depict the different test methodologies, both should elicit similar curves but in different x-axis units.

So why not impose more structure to the fitted parameters?
The simplest method of solving this, is to normalize the axis in relative terms.

Lets take a look at our function:

$$
    f(x) = a + b + \exp{\frac{x}{c}}
$$

To normalize this, we can do the following:

$$
    \begin{align}
        \hat{x} &= \frac{x}{x_{max}} \text{ and } \hat{c} = \frac{c}{x_{max}} \\
        \frac{x}{c} &= \frac{\hat{x}}{\hat{c}}
    \end{align}
$$

This will change x to the new domain:

$$
    \hat{x} \in [0,1]
$$

Which means to plot the new function we can just calculate within this range.

{% include scaled-curves.html %}
* After Normalization

