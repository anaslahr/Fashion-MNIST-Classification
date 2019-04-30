import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0


from keras.models import Model
from keras.layers import Input, Dense
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from matplotlib import pyplot as plt
import numpy as np

from keras.datasets import fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Training set (images) shape: {shape}".format(shape=train_images.shape))
print("Test set (images) shape: {shape}".format(shape=test_images.shape))

label_dict ={
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

plt.figure(figsize=[5,5])

plt.subplot(121)
curr_img = np.reshape(train_images[1],(28,28))
curr_lbl = train_labels[1]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl])+ ")")
plt.subplot(122)
curr_img = np.reshape(test_images[10],(28,28))
curr_lbl=test_labels[10]
plt.imshow(curr_img,cmap='gray')
plt.title("Label: " + str(label_dict[curr_lbl])+ ")")
plt.show()


#Rescaling the pixel values in range 0-1
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255

train_images = train_images.reshape(len(train_images), np.prod(train_images.shape[1:]))
test_images = test_images.reshape(len(test_images), np.prod(test_images.shape[1:]))
print(train_images.shape)
print(test_images.shape)


# train_images=train_images[0:5000]
# test_images=test_images[0:5000]


# Autoencoder parameters

batch_size=256
epochs=100


inChannel = 1
x,y = 28,28
input_img = Input(shape = (x,y,inChannel))
num_classes = 10

input_img= Input(shape=(784,))

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

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.summary()
encoder.summary()

autoencoder.compile(loss='binary_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])

autoencoder_train = autoencoder.fit(train_images,
                                    train_images,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_data=(test_images, test_images))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



encoded_imgs = encoder.predict(train_images)
predicted = autoencoder.predict(test_images)

plt.figure(figsize=(10, 4))
for i in range(5):
    # display original images
    ax = plt.subplot(3, 5, i +1)
    plt.imshow(test_images[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display encoded images
    ax = plt.subplot(3, 5, i + 5+1)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstructed images
    ax = plt.subplot(3, 5, 2 * 5 + i +1 )
    plt.imshow(predicted[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.savefig('deep_autoencoding.png')