#! /usr/bin/env python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# y = softmax(W * x + b)

# TensorFlow Session
sess = tf.Session()

def defineNetwork():
	global x, W, b, y
	
	# Input tensor
	x = tf.placeholder(tf.float32, [None, 784])

	# Weights and biases
	# Should I not set w with random small values?
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	# Output tensor
	y = tf.nn.softmax(tf.matmul(x, W) + b)


def train():
	global y_

	# Correct answers
	y_ = tf.placeholder(tf.float32, [None, 10])

	# Cost function (to minimize)
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))

	# Minimize cost function with Gradient Descent algo
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	
	# Initialize variables
	init = tf.initialize_all_variables()

	sess.run(init)

	# Train 1000 times with batches of 100
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

def evaluateModel():
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def main():
	defineNetwork()
	train()
	evaluateModel()


if __name__ == '__main__': main()

