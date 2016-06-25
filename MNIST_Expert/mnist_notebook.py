
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784]) # Input 
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # True Labels

W = tf.Variable(tf.zeros([784,10])) # Weights
b = tf.Variable(tf.zeros([10])) # Bias

# Prediction 
## y = Wx + b
y = tf.nn.softmax(tf.matmul(x, W) + b ) # Predicted Labels

# Cost Function 
## Cross entropy between the target (y) and model prediction (y_)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialise all variable and start session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# Run Training 
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





