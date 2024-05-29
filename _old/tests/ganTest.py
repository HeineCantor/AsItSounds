import keras

from keras import layers
from keras import ops
import tensorflow as tf
import numpy as np

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

# Dataset set-up

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

allDigits = np.concatenate([x_train, x_test])
allLabels = np.concatenate([y_train, y_test])

allDigits = np.float32(allDigits) / 255.0             # Normalization [0, 255] -> [0.0, 1.0]
allDigits = np.reshape(allDigits, (-1, 28, 28, 1))    # Adding a dimension: (#, 28, 28) -> (#, 28, 28, 1)

allLabels = keras.utils.to_categorical(allLabels, 10) # One-hot encoding: 2 -> [0 0 1 0 0 0 0 0 0 0 0]

dataset = tf.data.Dataset.from_tensor_slices((allDigits, allLabels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {allDigits.shape}")
print(f"Shape of training labels: {allLabels.shape}")

generatorInChannels = latent_dim + num_classes
discriminatorInChannels = num_channels + num_classes

print(f"Generator will have {generatorInChannels} channels")
print(f"Discriminator will have {discriminatorInChannels} channels")

# Model description (Generator + Discriminator)

discriminator = keras.Sequential(
    [
        layers.InputLayer((image_size, image_size, discriminatorInChannels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name = "discriminator",
)

generator = keras.Sequential(
    [
        layers.InputLayer((generatorInChannels,)),
        layers.Dense(7 * 7 * generatorInChannels),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, generatorInChannels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(num_channels, (7, 7), activation='sigmoid', padding='same'),
    ],
    name = "generator",
)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latend_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seedGenerator = keras.random.SeedGenerator(420)
        self.genLossTracker = keras.metrics.Mean(name="genLoss")
        self.discLossTracker = keras.metrics.Mean(name="discLoss")

    @property
    def metrics(self):
        return [self.genLossTracker, self.discLossTracker]
    
    def compile(self, dOptimizer, gOptimizer, loss_fn):
        super().compile()
        self.dOptimizer = dOptimizer
        self.gOptimizer = gOptimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        realImages, oneHotLabels = data

        imageOneHotLabels = oneHotLabels[:, :, None, None]  # Dummy dimension to concatenate labels to images
        imageOneHotLabels = ops.repeat(
            imageOneHotLabels, repeats=[image_size * image_size]
        )

        imageOneHotLabels = ops.reshape(
            imageOneHotLabels, (-1, image_size, image_size, num_classes)
        )

        batchSize = ops.shape(realImages)[0]
        randomLatentVectors = keras.random.normal(shape=(batchSize, latent_dim), seed=self.seedGenerator)
        randomVectorTables = ops.concatenate([randomLatentVectors, oneHotLabels], axis=1)

        generatedImages = self.generator(randomVectorTables)

        fakeImageAndLabels = ops.concatenate([generatedImages, imageOneHotLabels], axis=-1)
        realImageAndLabels = ops.concatenate([realImages, imageOneHotLabels], axis=-1)

        combinedImages = ops.concatenate([fakeImageAndLabels, realImageAndLabels], axis=0)

        labels = ops.concatenate([ops.ones((batchSize, 1)), ops.zeros((batchSize, 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combinedImages)
            dLoss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(dLoss, self.discriminator.trainable_weights)
        self.dOptimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        randomLatentVectors = keras.random.normal(shape=(batchSize, self.latent_dim), seed=self.seedGenerator)
        randomVectorLabels = ops.concatenate([randomLatentVectors, oneHotLabels], axis=1)

        misLeadingLabels = ops.zeros((batchSize, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            fakeImages = self.generator(randomVectorLabels)
            fakeImageAndLabels = ops.concatenate([fakeImages, imageOneHotLabels], axis=-1)
            predictions = self.discriminator(fakeImageAndLabels)
            gLoss = self.loss_fn(misLeadingLabels, predictions)

        gradients = tape.gradient(gLoss, self.generator.trainable_weights)
        self.gOptimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Update the metrics
        self.genLossTracker.update_state(gLoss)
        self.discLossTracker.update_state(dLoss)

        return {
            "g_loss": self.genLossTracker.result(),
            "d_loss": self.discLossTracker.result(),
        }
    
conditionalGAN = ConditionalGAN(discriminator, generator, latent_dim)

conditionalGAN.compile(
    dOptimizer=keras.optimizers.Adam(learning_rate=0.0003),
    gOptimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

conditionalGAN.fit(dataset, epochs=2)