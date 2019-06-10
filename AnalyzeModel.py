import sys
from keras import models
from keras.preprocessing import image
from keras import backend
from vis.visualization import visualize_activation,visualize_saliency,overlay,visualize_cam
from vis.utils import utils
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
        layer.output for layer in inception_model.layers[1:n_layers]]

    activation_model = models.Model(
        inputs=inception_model.layers[0].input, outputs=layer_outputs)

    return activation_model


def scale_img_values(channel_img):
    channel_img -= channel_img.mean()
    channel_img /= channel_img.std()
    channel_img *= 64
    channel_img += 128
    channel_img = np.clip(channel_img, 0, 255).astype('uint8')

    return channel_img


def analyze_channels_for_img(inception, activation_model, img_tensor,
                             n_layers=294, imgs_per_row=16, n_activations=16):
    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[0]
    print(first_layer_activation.shape)

    layer_names = [layer.name for layer in inception.layers[1:n_layers]]

    for layer_name, activation in zip(layer_names[:n_activations],
                                      activations[:n_activations]):
        if 'batch_normalization' in layer_name:
            continue

        # Channels in output
        n_channels = activation.shape[-1]

        img_size = activation.shape[1]
        # Columns to display in grid
        n_cols = n_channels // imgs_per_row

        display_grid = np.zeros((img_size * n_cols, imgs_per_row * img_size))

        for col in range(n_cols):
            for row in range(imgs_per_row):
                channel_img = scale_img_values(
                    activation[0, :, :, col * imgs_per_row + row])

                display_grid[col * img_size: (col + 1) * img_size,
                             row * img_size: (row + 1) * img_size] = channel_img

        scale = 1. / img_size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


def display_heatmap(model, img):
    layer_idx = utils.find_layer_idx(model, 'conv2d_15')
    heatmap = visualize_cam(model, layer_idx, filter_indices=None, seed_input=img[0,:,:,:])

    img_init = utils.load_img(test_img_path,target_size=(299,299))
    plt.imshow(overlay(img_init, heatmap))
    plt.show()


def display_filters(model, layer_names):
    for layer_name in layer_names:
        layer_idx = utils.find_layer_idx(model, layer_name)

        visu = visualize_activation(model, layer_idx, filter_indices=None)
        plt.imshow(visu)

    plt.show()


# Assumes the first layer of inputted model is inception.
def main(model_file):
    model = models.load_model(model_file)

    inception = model.layers[0]

    activation_model = get_activation_model(inception)
    img_tensor = get_img_tensor(test_img_path)

    analyze_channels_for_img(inception, activation_model, img_tensor)

    # These two functions should work but do not currently
    # display_heatmap(inception, img_tensor)

    layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
    # display_filters(inception, layer_names)



if len(sys.argv) != 2:
    print("Usage: python AnalyzeModel.py <model_file>")
else:
    main(sys.argv[1])
