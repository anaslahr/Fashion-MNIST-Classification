import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0

from TSNE import *
from SVM import *
from keras.models import Model
from keras.layers import Input, Dense

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist


def plot_original_imgs(train_images, test_images):
    plt.figure(figsize=[5, 5])

    plt.subplot(121)
    curr_img = np.reshape(train_images[1], (28, 28))
    curr_lbl = train_labels[1]
    plt.imshow(curr_img, cmap='gray')
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.subplot(122)
    curr_img = np.reshape(test_images[10], (28, 28))
    curr_lbl = test_labels[10]
    plt.imshow(curr_img, cmap='gray')
    plt.title("Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()


def rescale_pixel_values(train_images, test_images):
    # Rescaling the pixel values in range 0-1
    train_images_a = train_images.astype("float32") / 255
    test_images_a = test_images.astype("float32") / 255

    train_images_r = train_images_a.reshape(len(train_images_a), np.prod(train_images_a.shape[1:]))
    test_images_r = test_images_a.reshape(len(test_images_a), np.prod(test_images_a.shape[1:]))
    print(train_images_r.shape)
    print(test_images_r.shape)
    return train_images_r, test_images_r


def launch_autoencoder(epochs, train_images, train_labels, test_images, test_labels):
    # Autoencoder parameters

    batch_size = 256

    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape=(x, y, inChannel))
    num_classes = 10

    input_img = Input(shape=(784,))

    encoded = Dense(units=512, activation='relu')(input_img)
    encoded = Dense(units=256, activation='relu')(encoded)
    encoded = Dense(units=128, activation='relu')(encoded)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)
    decoded = Dense(units=64, activation='relu')(encoded)
    decoded = Dense(units=128, activation='relu')(encoded)
    decoded = Dense(units=256, activation='relu')(decoded)
    decoded = Dense(units=512, activation='relu')(decoded)
    decoded = Dense(units=784, activation='sigmoid')(decoded)

    auto_encoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    auto_encoder.summary()
    encoder.summary()

    auto_encoder.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    autoencoder_train = auto_encoder.fit(train_images,
                                         train_images,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         shuffle=True,
                                         validation_data=(test_images, test_images))

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(5)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    encoded_imgs = encoder.predict(train_images)
    predicted = auto_encoder.predict(test_images)

    return encoded_imgs, predicted


def plot_results(encoded_imgs, predicted):
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # display original images
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display encoded images
        ax = plt.subplot(3, 5, i + 5 + 1)
        plt.imshow(encoded_imgs[i].reshape(8, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(3, 5, 2 * 5 + i + 1)
        plt.imshow(predicted[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig('deep_autoencoding.png')


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[0:5000]
    test_images = test_images[0:1000]
    train_labels = train_labels[0:5000]
    test_labels = test_labels[0:1000]

    print("Training set (images) shape: {shape}".format(shape=train_images.shape))
    print("Test set (images) shape: {shape}".format(shape=test_images.shape))
    label_dict = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }
    plot_original_imgs(train_images, test_images)
    train_images_r, test_images_r = rescale_pixel_values(train_images, test_images)

    epochs = 5
    encoded_img, predicted = launch_autoencoder(epochs, train_images_r, train_labels, test_images_r, test_labels)
    plot_results(encoded_img, predicted);
    print(encoded_img.shape)
    call_tsne(50, encoded_img, train_labels)

    #call_svm(0.1,"poly",encoded_img,y_train,test_images,test_labels)