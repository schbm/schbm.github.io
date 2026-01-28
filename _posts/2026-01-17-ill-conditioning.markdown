---
layout: single
title:  "Ill-Conditioning in Linear Systems"
date:   2026-01-17 12:00:00 +0100
show_date: true
categories: ds 
tags: ds numerical-computation numeric-linear-algebra ill-conditioning norms
toc: true
---

As picked up in a [previous post]({% post_url 2025-06-25-addressing-common-challenges-dl-part1 %}),
we can observer that the ill conditioning of the hessian matrix
can pose problems in gradient optimization problems.

The goal of this post is to summarize and generalize this
problem to linear systems. We want to estimate how bad a problem is.

I want to define Ill-conditioning as shown in the book *Advanced Mathematics for Engineering* as:
> "(...) a computational problem is called ill-conditioned if small changes
in the data cause large changes in the solution. A problem is called
well-conditioned if small changes cause only small changes in the solution."

The quantitative factor on how bad this is depends on the problem at hand.

In Linear Systems:

$$
 Ax = b
$$

Ill-conditioning occurs if given two equations, those
lines are nearly parallel. Meaning if one moves substantially
the solution changes immensely.
This is also somehow true for larger systems, but the geometry no longer helps intuitively.

I tried to plot it here with a given linear system:

$$
\begin{aligned}
    0.9999x - 1.0001y &= 1\\
    x - y &= 1
\end{aligned}
$$

Produces solution $$ x=0.5, y=-0.5 $$.
If we now introduce small changes $$ \varepsilon $$:

$$
\begin{aligned}
    0.9999x - 1.0001y &= 1\\
    x - y &= 1 + \varepsilon
\end{aligned}
$$

This produces the solution $$ x = 0.5 + 5000.5 \varepsilon $$, $$ y=-0.5+4999.5 \varepsilon $$
and shows that for small epsilon a large change is produced of $$ 5000.5 \varepsilon $$.

I tried to plot it here. But since both lines are almost paralell they are not really distinguishable.
But you can easily see how a small change in epsilon affects the result a lot.
{% include near_singular_sensitivity.html %}
<p></p>

A widely used method to measure ill-conditioning is by the condition number
$$ k(A) $$ which is defined in  terms of norm.

# Norm

- A vector norm for column vector $$ x = [x_j] $$ with $$ n $$  components is a generalized length or distance
denoted by $$ \Vert x \Vert $$
- [Good Guide on Norm](https://builtin.com/data-science/vector-norms)



We define the **p-norm** as:

$$
    \lVert x \rVert_p = \left( \sum_{i=1}^{n} \lVert x_i \rVert^p \right)^{1/p}
$$


Usually one takes $$ p = 1 $$ (L1 Norm), $$ 2 $$ (Euclidian Norm) 
or the $$ ||x||_{\infty} $$ (Uniform Norm):


$$
   \lVert x \rVert_{\infty} = \underset{k}{max} \lVert x_k \lVert
$$

# Matrix Norm

Now for a matrix $$ A $$ with $$ n \times n $$
and vector $$ x $$ with $$ n $$ components it is provable that:

$$
    \Vert Ax \Vert \leqq c \Vert x \Vert
$$

That is there is a smallest possible value $$ c $$ depending on A,
which is called **matrix norm of A**
for all $$ x \neq 0 $$.

$$
    \Vert A \Vert = \max \frac{\Vert Ax \Vert}{\Vert x \Vert}
$$

By taking the smallest possible $$ c = \Vert A \Vert $$ we get:

$$
    \Vert Ax \Vert \leqq \Vert A \Vert \Vert x \Vert
$$

- The l1-norm of a matrix one gets the column "sum" norm.
(That is the biggest column sum norm)
- For the linfty-norm one gets the row "sum" norm
- For additional norms please advice google

# Condition Number of a Linear System

$$
    k(A) = \Vert A \Vert \Vert A^{-1} \Vert
$$

> A linear system of equations and its matrix $$ A $$whose condition number is small are well-conditioned. A large condition number indicates ill-conditioning.

In the before mentioned post, we apply additional tricks 
(or impose additional assumptions and conditions) to the matrix $$ A $$,
because computing the inverse of $$ A $$ is infeasable for large $$ A $$.
In deep learning most often $$ A $$ is huge.

![Neural Activation](/assets/images/2026-01-17-ill-conditioning/neural-activation.jpg)

