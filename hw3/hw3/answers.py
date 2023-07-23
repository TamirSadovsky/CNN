r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "I must attend his Majesty's command"
    temperature = 0.4
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

Splitting into sequences is memory efficient: Training on the entire text at once can take a lot of memory,
and splitting into smaller sequences allows you to load and process a batch of sequences at a time. This reduces memory requirements.

In addition, splitting into sequences mitigates the problem of vanishing gradients: When training on long sequences of data,
we are more prone to suffer from vanishing gradients problem. By splitting the corpus into smaller sequences, we can help
to mitigate the problem of vanishing gradients, which can lead to improved model performance.

Furthermore, splitting the data into sequences speed up the training: It  allows us to train with batches, which means
that we can process multiple sequences simultaneously, which improves the training efficiency and enables parallelization
in the training. 

"""

part1_q2 = r"""
**Your answer:**

It is possible that the generated text shows memory longer than the sequence length, because we use the hidden state and
we update it at each time-step. The hidden state acts as a form of memory, allowing the network to capture dependencies over time.
This is why we can keep using the hidden state to create text longer than the sequence length.

"""

part1_q3 = r"""
**Your answer:**

We are not shuffling the order of the batches when training, because each batch has chars in a specific order. The order
of the chars is important to create the sentences in text. We don't want to change the order of the chars, so that the
sequence will be correct. 

"""

part1_q4 = r"""
**Your answer:**

1. We use lower the temperature for sampling, because lowering the temperature allows more control on the generated text. 
Higher temperature produces more diverse output but can be less coherent. In contrast, lower temperature gives more
deterministic output that matches more the patterns learned during training, but less diverse. In training, we use high
temperature because it adds more randomness into the training. This encourages exploration of different possibilities
and prevents the model from getting stuck in a local minimum. In sampling, we use lower temperature because we want to
control the randomness and generate more deterministic output.  

2. When the temperature is very high the output is diverse because the distribution becomes more uniform. So when the model
generates a char it chooses it is from a more uniform distribution so it might chooses other plausible results.

3. When the temperature is very low the output is deterministic that matches more the patterns learned during training.
This is because the distribution becomes less uniform. So when the model generates a char it tends to choose the most
probable options based on the learned patterns during training.

"""
# ==============


# ==============
# Part 2 answers


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32,
        h_dim=1024,
        z_dim=4,
        x_sigma2=5,
        learn_rate=0.0005,
        betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""

The hyperparameter $\sigma^2$ acts as the variance of the Gaussian distribution and serves as a regularization factor for the data_loss component of the loss function.
By adjusting the value of $\sigma^2$, we can control the significance of the data loss relative to the KL-divergence loss. 
A larger $\sigma^2$ results in a less significant data loss, allowing for a balance between the two loss terms.

"""

part2_q2 = r"""

1. The reconstruction loss measures how well the VAE can recreate the original input, 
while the KL divergence loss helps shape the structure of the latent space.

2. The KL loss term influences how the data is distributed in the latent space, 
encouraging it to follow a desired distribution pattern, often a standard Gaussian. 
This promotes a more organized and useful representation.
The KL loss term acts as a regularization mechanism that prevents overfitting and helps the model generalize better to unseen data.

3. Using the KL divergence loss in VAE has a benefit of helping us find the right balance between how similar the reconstructed data is to the original input and how well-organized the latent space is.
By changing the weight of the KL term, 
we can control how much we prioritize accurate reconstruction versus creating a latent space that is organized and meaningful,
giving us the flexibility to capture important features.


"""

# ==============


# ==============
# Part 3 answers


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
