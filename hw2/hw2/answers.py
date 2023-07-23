r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.1
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        wstd = 0.1
        lr = 0.01
        reg = 0.001
    if opt_name == 'momentum':
        wstd = 0.1
        lr = 0.01
        reg = 0.001
    if opt_name == 'rmsprop':
        wstd = 0.1
        lr = 0.0001
        reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
My expectation for the train accuracy was to see it drop lower for higher dropout values - Since 
the model has less information in the training part, and can't over-fit the training data.
My expectation for the test accuracy however was to see the best accuracy for dropout=0.4 - Because
when dropout=0 the model over-fits, and when dropout=0.8 the model under-fits (not learning enough 
since it losses 80% of neurons for each layer). 

That expectation was met by the results of the graph.
"""

part2_q2 = r"""
Yes, it is possible, Because the accuracy is summing up discrete values, while the cross entropy loss sums up (with a log transformation) continous values.

For Example - Say I have 2 samples from 2 possible classes, and look at this two cases:
    * I predict both sample INCORRECTLY, with a confidence level of p=0.5 for each (guessed wrong both times...). The accuracy is 0, and the loss is -2*log(0.5) ~= 1.4
    * I predict on sample CORRECTLY and one INCORRECTLY, with a confidence level of p=0.9 for each. The accuracy is 0.5, and the loss is -(log(0.1) + log(0.9)) ~= 2.4
    
This is because accuracy represents if I'm right, and cross entropy loss represents how "bad" my error were.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

The experimental results indicate that increasing the depth of the network leads to a decrease in model accuracy.
The best performance was achieved with a network depth of 4.
However, we encountered challenges when attempting to train a network with a depth of 16 due to the presence of MaxPooling layers, which caused significant dimensionality reduction.

To address this issue, a possible solution would be to reduce the network depth or limit the usage of MaxPooling layers. 
This approach aims to mitigate the impact of dimensionality reduction and improve the network's training capabilities.

"""

part3_q2 = r"""
<!--# Analyze your results from experiment 1.2. In particular, compare to the results of experiment 1.1. -->

The impact of varying the number of convolutional layers ($L$) on our model's complexity, considering $L$ as 2, 4, and 8:
As the number of convolutional layers increased, the train accuracy also increased. This effect is particularly pronounced in the case of $L=8$.
It is important to note that such high accuracy levels can be attributed to overfitting on the training data.

We explored the influence of varying $K$ (the number of filters) on the model's behavior. We observed that as $K$ increased, the onset of overfitting occurred earlier.
This observation is evident in the train loss graph, where a significant drop in the train loss is observed. However, these values performed better than others until overfitting began.
We saw that adding more convolutional layers and increasing the number of filters can improve model accuracy up to a certain point.
However, we need to be careful about overfitting as we continue to increase these parameters.
"""

part3_q3 = r"""
<!-- # Analyze your results from experiment 1.3. -->

As we examine the impact of increasing the value of $L$ on our model, we observe an improvement in performance. The test accuracy graph demonstrates that when $L=4$, we achieve the best results.
Furthermore, an interesting observation is that as we increase the value of $L$, we notice a delay in the onset of overfitting, leading to better overall results. This suggests that a higher value of $L$ allows the model to generalize better to unseen data.

In summary, increasing the value of $L$ with varying numbers of convolutional layers proves beneficial for our model's performance. 
The test accuracy and delayed overfitting indicate the advantages of employing a higher value of $L$ in our network.



"""


part3_q4 = r"""
<!-- Explain your modifications to the architecture which you implemented in the YourCodeNet class. -->
<!-- Analyze the results of experiment 2. Compare to experiment 1. -->

We choose the implementation used model VGG13, deep convolutional neural network known for its strong performance.
Through our experiments, we have achieved great results with VGG13 model, surpassing expectations associated with such a renowned architecture. 
Specifically, we have achieved a test accuracy of over 80%, demonstrating the effectiveness and robustness of our approach. 
These results highlight the capability of the VGG13-based model to excel in various tasks and validate its reputation as a powerful and reliable architecture in the field of deep learning.




"""
