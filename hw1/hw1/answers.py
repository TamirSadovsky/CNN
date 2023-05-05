r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
No, increasing $k$ does not necessarily lead to improved generalization for unseen data. In many cases, it may lead to underfitting, causing the model to become too simple and unable to capture patterns in the data. The optimal $k$ value depends on the dataset and the given problem, and should be determined through cross-validation. 
If we set $k$=1 in kNN model, it will only consider the closest neighbor to the test sample for making a prediction. This can lead to overfitting because it may only work well on the training data and may not generalize well to unseen data. On the other hand, if we set a very large value for $k$, the model will consider a lot of neighbors, possibly including irrelevant ones, and may result in underfitting. Therefore, finding the optimal value of $k$ depends on the dataset and the specific problem.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Linearity allows for arbitrary scaling of $\Delta$. If $W_1$ is a solution that minimizes the loss function, then $\alpha \cdot W$ (where $0 < \alpha \in \mathbb{R}$) also minimizes the loss function. The solution remains the same, but the objective value is scaled. We can choose any $\Delta$, and the loss function will be minimized with corresponding weights $W$. We can always find a constant to represent the same hyperplane.

We have a minimization problem where $L_i(W)=\sum_{i\not=y_i}max(0,\Delta+w_j^\top x_i-w_{y_i}^\top x_i)$ and we want to minimize $L(W)$. We can scale the weights by $c>0$ and if we scale $\Delta$ by $c>0$ as well, we get a scaled version of the objective function. However, the scaled objective function has the same solution as the unscaled version. Therefore, we can scale the regularization term by $c$ as well, and the problem remains the same.

"""
part3_q2 = r"""
1. The linear model is learning to separate the two classes in the dataset using a linear decision boundary. The decision boundary is represented by the line in the visualization. Points above the line are classified as one class, while points below the line are classified as the other class.

Some of the classification errors in the linear model occur when the points are located near the decision boundary. In these cases, the linear model may not be able to confidently determine the correct class, leading to misclassifications.

2. The interpretation of the linear model is different from KNN in that the linear model is learning a decision boundary that separates the classes based on a weighted combination of the input features, while KNN is making predictions based on the similarity between new points and the training data. In other words, the linear model is learning a generalization of the training data, while KNN is using the training data directly to make predictions. However, both models are used for classification tasks and can be effective in different situations.


"""

part3_q3 = r"""
1. The learning rate is deemed to be "good" as the loss shows a noticeable decrease after just a few epochs, which would not be observed with a low learning rate. Conversely, an excessive learning rate would result in an escalating loss.

2. Regarding the graph of the training and test set accuracy, I would say that the model is "slightly overfitted to the training set".

If the model were highly overfitted to the training set, we would see a very high accuracy on the training set, but a significantly lower accuracy on the test set. On the other hand, if the model were highly underfitted to the training set, we would see a low accuracy on both the training and test sets. However, in this case, we can see that the accuracy on the training set is very high, close to 100%, but the accuracy on the test set is slightly lower, around 85-90%. This suggests that the model is fitting well to the training set, but it may not be generalizing well to new examples, which is a sign of slight overfitting.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
A residual plot with a straight line of 0 is considered an ideal pattern. However, in realistic cases, a randomly distributed points around the 0-line from both sides is more common. Each improvement made to the model, such as feature engineering or cross-validation for parameter selection, led to an improved residual plot. As the error reduced, the points on the plot became closer to the 0-line, indicating that the fitness of the model improved with each change.
"""

part4_q2 = r"""

1. We used np.logspace instead of np.linspace when defining the range for lambda in the cross-validation code, as it is better to sample values from different orders of magnitude. Sampling in logspace allows for better coverage of the scale, as we do not know if the regularization term should be very small, medium-sized or large.
2. Our hyperparameters are $\lambda$ and $degree$ of the polynomial features, with 20 values for $\lambda$ and 3 values for $degree$. We used sklearn.model_selection.GridSearchCV to try all possible combinations of the hyperparameters with 3-fold cross-validation, which totals to $20 x 3 x 3 = 180$ combinations. Our model was fitted 180 times.
"""

# ==============
