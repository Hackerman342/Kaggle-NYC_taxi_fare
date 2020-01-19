#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Establish environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import os


# In[77]:


# Load training and test data (max nrows = 55M)
raw_train =  pd.read_csv('input_data/train.csv', nrows = 2_000_000)
raw_test =  pd.read_csv('input_data/test.csv', nrows = 100_000)


# In[78]:


# Set data pre-processing parameters
max_fare = 200
max_seats = 8
max_dist = 1


# In[79]:


# Define useful functions
def norm(x):
    return (x -x.mean()) / x.std()


# In[83]:


# Preprocess training data
train = raw_train

# Remove any data with invalid entries
train = train.dropna(how = 'any', axis = 'rows')

# Extract year, hour, and city-block distance
train = train.assign(year = train.pickup_datetime.str[:4].astype(int))
train = train.assign(hour = train.pickup_datetime.str[11:13].astype(int))
train = train.assign(distance =  abs(train.dropoff_latitude-train.pickup_latitude) + abs(train.dropoff_longitude-train.pickup_longitude))

# Remove outliers
train = train[(train.fare_amount > 0) & (train.fare_amount <= max_fare)]
train = train[(train.passenger_count > 0) & (train.passenger_count <= max_seats)]
train = train[(train.distance < max_dist)]

# Separate input & output variables
#x_train = train[['passenger_count', 'year', 'hour', 'distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
x_train = train[['passenger_count', 'year', 'hour', 'distance']]
y_train = train[['fare_amount']]

# Normalize training data set
x_train = norm(x_train)


# In[85]:


## Multiple layers
tf.reset_default_graph() 
# Network Parameters
learning_rate = 0.001
epochs = 50
print_steps = 5

n_samples = x_train.shape[0]
n_params = x_train.shape[1]

# Tensor placeholders
X = tf.placeholder(tf.float32, shape = [None, n_params])
Y = tf.placeholder(tf.float32, shape = [None, 1])

# Model variables
W1 = tf.get_variable("weight1", shape=[n_params, 7],
                    initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable("bias1", shape=[7],
                    initializer=tf.constant_initializer(0.1))

W2 = tf.get_variable("weight2", shape=[7, 17],
                    initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable("bias2", shape=[17],
                    initializer=tf.constant_initializer(0.1))

W3 = tf.get_variable("weight3", shape=[17, 21],
                    initializer=tf.glorot_uniform_initializer())
b3 = tf.get_variable("bias3", shape=[21],
                    initializer=tf.constant_initializer(0.1))

W4 = tf.get_variable("weight4", shape=[21, 11],
                    initializer=tf.glorot_uniform_initializer())
b4 = tf.get_variable("bias4", shape=[11],
                    initializer=tf.constant_initializer(0.1))

W5 = tf.get_variable("weight5", shape=[11, 1],
                    initializer=tf.glorot_uniform_initializer())
b5 = tf.get_variable("bias5", shape=[1],
                    initializer=tf.constant_initializer(0.1))

# Initialize the variables
init = tf.global_variables_initializer()

# Linear model
Y1 = tf.add(tf.matmul(X, W1), b1)
Y2 = tf.add(tf.matmul(Y1, W2), b2)
Y3 = tf.add(tf.matmul(Y2, W3), b3)
Y4 = tf.add(tf.matmul(Y3, W4), b4)
Y5 = tf.add(tf.matmul(Y4, W5), b5)
pred = Y5

# Mean squared error
cost = tf.reduce_mean(tf.square(pred-Y))
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})

        # Display log at given epoch interval
        if (epoch+1) % print_steps == 0:
            c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            

    print("Optimization Complete!")
    
    training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    print("Mean square error =", training_cost)


# In[ ]:




