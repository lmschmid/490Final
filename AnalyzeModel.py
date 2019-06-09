import sys
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


test_img_path = 'data/train/a21caab32c00011d38f1e409fbf65d19.jpg'


def get_img_tensor(img_path, display_img=False):
    img = image.load_img(img_path, target_size=(299, 299))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if display_img:
        plt.imshow(img_tensor[0])
        plt.show()

    return img_tensor


# Returns model which returns output of first n_layers given an image.
def get_activation_model(inception_model, n_layers=294):
    layer_outputs = [
        layer.output for layer in inception_model.layers[1:294]]

    activation_model = models.Model(
        inputs=inception_model.layers[0].input, outputs=layer_outputs)

    return activation_model


def analyze_img(activation_model, img_tensor):
    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[0]
    print(first_layer_activation.shape)

    plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
    plt.show()


# Assumes the first layer of inputted model is inception.
def main(model_file):
    model = models.load_model(model_file)

    inception = model.layers[0]

    activation_model = get_activation_model(inception)
    img_tensor = get_img_tensor(test_img_path)

    analyze_img(activation_model, img_tensor)


if len(sys.argv) != 2:
    print("Usage: python AnalyzeModel.py <model_file>")
else:
    main(sys.argv[1])
