# Text-Categorizer

This project uses a recurrent neural network for a text categorization application. It uses an LSTM architecture in order to mitigate the vanishing / exploding gradient problem.

It uses the Tensorflow library to build the model.

The dataset I used is corpus1. Since the dataset was small for deep learning standards, the results were not as accurate as the Rocchio / TF\*IDF method, in Project 1.

The first architecture I came up with was Embedding -> Bidirectional LSTM -> Dense -> Dense. I also tried LSTM layer instead of Bidirectional LSTM to test its performance, and the bidirectional layer performed significantly better.

A dropout layer was added to mitigate overfitting (initially with a rate of 0.5). To my surprise, increasing the dropout rate to 0.6 performed better.

The final architecture was Embedding -> Bidirectional LSTM -> Dense -> Dropout -> Dense

# Dividing Training Tuning, Testing Set

The training/tuning set was divided by a 9:1 ratio. 9:1 ratio consistently performed better than 8:2 ratio. The testing set
was kept as initially provided.

Train set (885 training articles) & 443 testing articles

# Tested parameters/ Hyperparameters:

Activation
maxlen of input data
vocab_size
embedding dimension
num_epochs

Out of tanh, sigmoid, and relu, relu performed best.
The maximum length of the input data was initially set to 350, since the average article was approx. 370 words long. However, some articles were 50 words long, and others were 400-500 words long. Lowering the maximum length to around 250 improved the results.
The vocabulary size did not seem to have as big an impact as the maximum length hyperparameter, but performed worse when it was too low (1000-3000) or too high (>7000). I ran multiple tests, and 5000 performed consistently well.
For the dimension of the layer, 32, 64, and 128 were tested. 64 performed best.
A lot of num_epochs were tested. 10, 11, 20, 30, 40. I thought running 30-40 epochs would lead to overfitting, but surprisingly performed consistently better at 30~40 epochs. 30 epochs were kept because it seemed to have similar results to 40 epochs.

Padding and truncating was set to "pre" as default, but "post" seemed to perform better, although I found articles saying "pre" usually performs better for these tasks.

# Final Settings

model = tf.keras.Sequential([tf.keras.layers.Embedding(dict_size, embedding_dim),
tf.keras.layers.Bidirectional(
tf.keras.layers.LSTM(embedding_dim)),
tf.keras.layers.Dense(embedding_dim, activation = 'relu'),
tf.keras.layers.Dropout(0.6),
tf.keras.layers.Dense(6, activation='softmax')])

Training / Tuning ratio: 9:1
vocab_size: 5000
max_len: 256
embedding_dimension: 64
num_epochs: 30
padding & trunc: post

# Results:

The highest score obtained was around ~89%, but the average was around 83~85%. It performed worse than the Rocchio/TF\*IDF, but I believe it may be because the dataset is quite small for NLP standards and the results were highly inconsistent on every iteration
