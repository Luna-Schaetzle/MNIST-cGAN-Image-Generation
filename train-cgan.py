import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ===========================
# 1. Daten vorbereiten
# ===========================

def load_and_preprocess_data():
    # Laden des MNIST-Datensatzes
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # Normalisieren der Bilder auf den Bereich [-1, 1]
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5

    # Hinzufügen einer Kanal-Dimension (für Graustufenbilder)
    train_images = np.expand_dims(train_images, axis=-1)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 128

    # One-Hot-Encoding der Labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

    # Erstellen eines TensorFlow-Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset

# ===========================
# 2. Generator erstellen
# ===========================

def make_generator_model():
    noise_dim = 100
    label_dim = 10

    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(label_dim,))

    # Kombinieren von Rauschen und Label
    merged_input = layers.Concatenate()([noise_input, label_input])

    x = layers.Dense(7*7*256, use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh')(x)

    generator = models.Model([noise_input, label_input], x, name="Generator")
    return generator

# ===========================
# 3. Diskriminator erstellen
# ===========================

def make_discriminator_model():
    image_shape = (28, 28, 1)
    label_dim = 10

    image_input = layers.Input(shape=image_shape)
    label_input = layers.Input(shape=(label_dim,))

    # Label auf die Bildgröße erweitern und concatenaten
    label_embedding = layers.Dense(np.prod(image_shape))(label_input)
    label_embedding = layers.Reshape(image_shape)(label_embedding)

    merged_input = layers.Concatenate()([image_input, label_embedding])

    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(merged_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = models.Model([image_input, label_input], x, name="Discriminator")
    return discriminator

# ===========================
# 4. Verlustfunktionen und Optimierer definieren
# ===========================

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # Echte Bilder sind 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Falsche Bilder sind 0
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # Generator will fake_output als echt sehen

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ===========================
# 5. Trainingsschritt definieren
# ===========================

@tf.function
def train_step(images, labels, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise_dim = 100
    batch_size = images.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# ===========================
# 6. Bildgenerator zum Speichern der Bilder
# ===========================

def generate_and_save_images(model, epoch, test_noise, test_labels, output_dir='generated_images'):
    predictions = model([test_noise, test_labels], training=False)

    # Denormalisieren von [-1,1] zu [0,1]
    predictions = (predictions + 1) / 2.0

    # Erstellen des Ausgabeverzeichnisses, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 1))

    for i in range(predictions.shape[0]):
        plt.subplot(1, 10, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# ===========================
# 7. Checkpointing
# ===========================

def setup_checkpointing(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir='./training_checkpoints'):
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    return checkpoint, checkpoint_prefix

# ===========================
# 8. Training Loop
# ===========================

def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, seed_noise, seed_labels, checkpoint, checkpoint_prefix, output_dir='generated_images'):
    for epoch in range(1, epochs + 1):
        start = time.time()

        gen_loss_list = []
        disc_loss_list = []

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

        # Jede Epoche ein Bild generieren
        generate_and_save_images(generator, epoch, seed_noise, seed_labels, output_dir=output_dir)

        # Checkpoints speichern
        checkpoint.save(file_prefix = checkpoint_prefix)

        print (f'Epoch {epoch}/{epochs} - Gen Loss: {np.mean(gen_loss_list):.4f}, Disc Loss: {np.mean(disc_loss_list):.4f} - Zeit: {time.time()-start:.2f} sec')

    # Letzte Bilder nach dem Training
    generate_and_save_images(generator, epochs, seed_noise, seed_labels, output_dir=output_dir)

# ===========================
# 9. Hauptfunktion
# ===========================

def main():
    # Parameter
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 10
    output_dir = 'generated_images_cgan'

    # Seed für die Konsistenz der Bilder während des Trainings
    seed_noise = tf.random.normal([num_examples_to_generate, noise_dim])
    seed_labels = tf.one_hot(np.arange(10), depth=10)  # Labels 0-9

    # Daten vorbereiten
    train_dataset = load_and_preprocess_data()

    # Modelle erstellen
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Checkpointing einrichten
    checkpoint, checkpoint_prefix = setup_checkpointing(generator, discriminator, generator_optimizer, discriminator_optimizer)

    # Training starten
    train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, seed_noise, seed_labels, checkpoint, checkpoint_prefix, output_dir=output_dir)

    # Generator-Modell speichern
    generator.save('generator_cgan.h5')
    print(f'Training abgeschlossen. Generator-Modell gespeichert als "generator_cgan.h5". Generierte Bilder sind im Ordner "{output_dir}" gespeichert.')

# ===========================
# 10. Skript ausführen
# ===========================

if __name__ == '__main__':
    main()
