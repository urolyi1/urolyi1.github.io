---
layout: project
title:  "Neural Net from Scratch"
date:   2019-09-09 20:50:00 -0400
proj_num: 3
mathjax: true
categories: project machine-learning numpy
---

Recently, I implemented a neural network using only numpy arrays. This was something I was curious about ever since I watched the 3Blue1Brown series on backpropagation. Even though the code itself is short this was quite a challenging task. Understanding the theory and what backpropagation does at an intuitive level is different than actually implementing it while making sure all the linear algebra works. There are a few things I will add to the code soon like different types of transition functions and some demos. However, the base class works and is on [github][github-link].

I will now just briefly describe the key equations needed to implement this neural network correctly. For forward propagation, 
\begin{equation}
a^l = \sigma^l(z^l), z^l = a^{l-1}W^l + b^l
\end{equation}
Here $a^l$ is the output of layer $l$, $z^l$ is the input to the transition function, $\sigma^l$, and $W^l$ and $b_l$ are the weight and bias matrix for layer $l$, respectively. This is pretty straightforward and is just matrix multiplication of the previous layer's output with the weights added to the bias then put through the transition function. The backpropagation is where your head can really start to spin. I'm not going to really go into the theory behind backpropagation in this post, but all you need to know is that you want to move in the opposite direction of the gradient of the loss function. To do this we have to calculate the gradient at each epoch.

$$
\begin{eqnarray}
\partial z^l &=& \begin{cases} a^l - y & l = L\\ \partial z^{l+1} (W^{l+1})^T \odot \sigma^\prime(z^l) & 0\leq l < L \end{cases} 
	\\
    \partial W^l &=& a^{l-1} \partial z^l
    \\
    \partial b^l &=& \partial z^l
\end{eqnarray}
$$

There is a lot to unpack here but $W^l$ is the gradient for the layer $l$ weights matrix and will be subtracted from the current weights matrix, $y$ is the true value and $\sigma^\prime$ is the derivative of the transition function, and $\odot$ is the hadamard product operator which is just an element-wise product. One of the main challenges of this project was just making sure all the dimensions were correct and figuring out where I needed transposes etc. I will soon be adding the bias factor to the neural net, $b$ and also I'm interested in adding some additional optimizations especially to the step size, so this post might be updated.


[github-link]: https://github.com/urolyi1/NeuralNetwork