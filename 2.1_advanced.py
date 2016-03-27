#! /usr/bin/env python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
def conv2d(x, W):
	return tf.nn.conv2d(
		x,
		W,
		strides = [1, 1, 1, 1],
		padding = 'SAME'
	)

def max_pool_2x2(x):
	return tf.nn.max_pool(
		x,
		ksize   = [1, 2, 2, 1],
		strides = [1, 2, 2, 1],
		padding = 'SAME'
	)

def main():
	# Placeholders
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# Variables
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	
	# Initialize variables
	sess.run(tf.initialize_all_variables())
	
	# Model Definition
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	
	# Cost function
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	
	# Train step
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	
	# Train 1000 times
	for i in range(1000):
		batch = mnist.train.next_batch(50)
		
		train_step.run(
			feed_dict = {
				x  : batch[0],
				y_ : batch[1]
			}
		)
	
	# Evaluate model
	correct_prediction = tf.equal(
		tf.argmax(y,  1),
		tf.argmax(y_, 1)
	)
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	print (
		accuracy.eval(
			feed_dict = {
				x  : mnist.test.images,
				y_ : mnist.test.labels
			}
		)
	)


if __name__ == '__main__': main()

