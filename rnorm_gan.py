# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:41:05 2021

@author: iverm
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy

'''
generator
'''
INPUT_DIM = 4

class Generator(Model):
    def __init__(self):
        super().__init__()
        
        self.dense2 = Dense(20, activation='linear')
        self.dense3 = Dense(2, activation='linear')
        
    def call(self, z):
        z = self.dense2(z)
        z = self.dense3(z)
        return z

G = Generator()

'''
discriminator
'''
class Discriminator(Model):
    def __init__(self):
        super().__init__()
        
        self.dense1 = Dense(100, activation='relu')
        self.dense2 = Dense(20, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')
        
    def call(self, x):
        
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

D = Discriminator()

'''
GAN
'''
def generator_loss(z):
    return binary_crossentropy(tf.ones_like(z), z)

def discriminator_loss(x, z):

    return (binary_crossentropy(tf.ones_like(x), x) +
            binary_crossentropy(tf.zeros_like(z), z))

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x):
    noise = np.random.uniform(size = (x.shape[0], INPUT_DIM)) # size=x.shape
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        z = G(noise)
        
        real_output = D(x)
        fake_output = D(z)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, G.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)
    
    generator_optimizer.apply_gradients(
        zip(gen_gradients, G.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, D.trainable_variables))

    return gen_loss, disc_loss        

    
def train(x, epochs):
    gen_loss = np.zeros(epochs)
    disc_loss = np.zeros(epochs)

    for epoch in range(epochs):
        #x = np.random.multivariate_normal(mu, sigma, size=1000)
        a, b = train_step(x)

        gen_loss[epoch] = tf.math.reduce_mean(a)
        disc_loss[epoch] = tf.math.reduce_mean(b)

    return gen_loss, disc_loss
        
# parameters for true distribution
mu = np.array([5.1, 1.7])
sigma = np.array([2, 0.3, 0.3, 2]).reshape(2, 2)

BATCH_SIZE = 1000
x = np.random.multivariate_normal(mu, sigma, size=BATCH_SIZE)

# train network
EPOCHS = 10000
G_loss, D_loss = train(x, EPOCHS)

# simulating from generator
SAMPLE_SIZE = 500
rnorm = G(np.random.uniform(size=(SAMPLE_SIZE, INPUT_DIM)))

# print accuracy
print('Discriminator accuracy for true samples: '
      f'{np.sum(np.round(D(x))) / 1000}') 
print('Discriminator accuracy for false samples: '
      f'{np.sum(np.round(D(rnorm)) == 0) / 1000}')

# plot results
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(G_loss, label='Generator loss')
ax1.plot(D_loss, label='Discriminator loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()    
ax2.scatter(x[:, 0], x[:, 1], label='Training distribution')
ax2.scatter(rnorm[:, 0], rnorm[:, 1], label='Generator distribution')
ax2.legend()