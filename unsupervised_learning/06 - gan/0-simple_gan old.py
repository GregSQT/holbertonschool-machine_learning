#!/usr/bin/env python3
# simple_gan.py
# Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
# nisi non justo accumsan auctor. Maecenas molestie orci a erat
# faucibus tempus. Quisque quis odio nibh. Sed ac turpis ante.
# Duis sit amet feugiat massa, non maximus augue. Phasellus bibendum
# ultrices lacinia. In auctor risus at faucibus ultrices. Aenean convallis
# sit amet ipsum a commodo. Integer pulvinar maximus lorem et volutpat.
# estibulum orci libero, sagittis in pulvinar iaculis, volutpat eu arcu.
import tensorflow as tf
from tensorflow import keras
import numpy as np


# creation of the simple_GAN class
# Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
class Simple_GAN(keras.Model):
    # Init
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()  # Initialize Keras.Model
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Generator loss and optimizer
        # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
        self.generator.loss = lambda x: \
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = \
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # Discriminator loss and optimizer
        # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
        self.discriminator.loss = lambda x, y: (
                tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
                tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # Generate real samples
    # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Generate fake samples
    # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Perform one training step
    # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(real_output, fake_output)

            discr_grads = tape.gradient(discr_loss,
                                        self.discriminator.trainable_variables)
            self.discriminator.optimizer.\
                apply_gradients(zip(discr_grads,
                                    self.discriminator.trainable_variables))

        # Boucle while
        # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_output = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(gen_output)

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.\
            apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        # return
        # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vel
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
