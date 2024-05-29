import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from keras import layers
from keras import ops

from music21 import converter, instrument, note, chord, stream
from pathlib import Path

SEQUENCE_LENGTH = 100
LATENT_DIM = 1000
BATCH_SIZE = 16
EPOCHS = 100
SAMPLE_INTERVAL = 1

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = LATENT_DIM
        self.disc_loss = []
        self.gen_loss = []

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        self.discriminator = self.get_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.get_generator()

        z = layers.Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(generated_seq)

        self.combined = keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_discriminator(self):
        model = keras.Sequential(
            [
                layers.LSTM(512, input_shape=self.seq_shape, return_sequences=True),
                layers.Bidirectional(layers.LSTM(512)),
                layers.Dense(512),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(256),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(100),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ],
            name = "discriminator",
        )

        model.summary()

        sequence = layers.Input(shape=self.seq_shape)
        validity = model(sequence)

        return keras.Model(sequence, validity)

    def get_generator(self):
        model = keras.Sequential(
            [
                layers.Dense(256, input_dim=self.latent_dim),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(512),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(1024),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(np.prod(self.seq_shape), activation='tanh'),
                layers.Reshape(self.seq_shape)
            ],
            name = "generator",
        )

        model.summary()

        noise = layers.Input(shape=(self.latent_dim,))
        sequence = model(noise)

        return keras.Model(noise, sequence)
    
    def train(self, epochs, batch_size=128, sample_interval=50):
        notes = get_notes()
        n_vocab = len(set(notes))

        X_train, y_train = prepare_sequences(notes, n_vocab)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
                self.disc_loss.append(d_loss[0])
                self.gen_loss.append(g_loss)

        self.generate(notes)
        self.plot_loss()

    def generate(self, input_notes):
        # Get pitch names and store in a dictionary
        notes = input_notes
        pitchnames = sorted(set(item for item in notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        
        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        
        pred_notes = [x*242+242 for x in predictions[0]]
        
        # Map generated integer indices to note names, with error handling
        pred_notes_mapped = []
        for x in pred_notes:
            index = int(x)
            if index in int_to_note:
                pred_notes_mapped.append(int_to_note[index])
            else:
                # Fallback mechanism: Choose a default note when the index is out of range
                pred_notes_mapped.append('C5')  # You can choose any default note here
        
        create_midi(pred_notes_mapped, 'gan_final')

        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

'''
    get_notes: Function to get notes from a dataset
    @return: A list of notes
'''
def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []

    for file in Path("archive").glob("*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab) / 2) / (float(n_vocab) / 2)
    network_output = keras.utils.to_categorical(network_output, num_classes=n_vocab)  # Use to_categorical from TensorFlow's Keras

    return network_input, network_output  # Add this return statement

def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for item in prediction_output:
        pattern = item[0]
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

if __name__ == '__main__':
    gan = GAN(rows=SEQUENCE_LENGTH)    
    gan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)

    # Save the generator and discriminator models
    gan.generator.save("generator_model.h5")
    gan.discriminator.save("discriminator_model.h5")