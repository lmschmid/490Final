import sys
from keras import models
from keras.preprocessing import image
from keras import backend
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


# Credit: Chollet
def analyze_layer_filter(inception, layer_name, filter_index=0):
    layer_output = inception.get_layer(layer_name).output

    loss = backend.mean(layer_output[:, :, :, filter_index])

    # The call to `gradients` returns a list of tensors (of size 1 in this case)
    # hence we only keep the first element -- which is a tensor.
    grads = backend.gradients(loss, inception.layers[0].input)[0]
    # We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
    grads /= (backend.sqrt(backend.mean(backend.square(grads))) + 1e-5)

    iterate = backend.function([inception.layers[0].input], [loss, grads])

    # Gray image with noise.
    input_img_data = np.random.random((1, 299, 299, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.  # this is the magnitude of each gradient update
    for _ in range(40):
        # Compute the loss value and gradient value
        _, grads_value = iterate([input_img_data])
        # Here we adjust the input image in the direction that maximizes the loss
        input_img_data += grads_value * step

    img = input_img_data[0]

    return  scale_img_values(img)



def analyze_filters(inception, layer_names):
    for name in layer_names:        
        grid = analyze_layer_filter(inception, name)
        plt.title(name)
        plt.grid(False)
        plt.imshow(grid)

        plt.show()


# Assumes the first layer of inputted model is inception.
def main(model_file):
    model = models.load_model(model_file)

    inception = model.layers[0]

    activation_model = get_activation_model(inception)
    img_tensor = get_img_tensor(test_img_path)

    # analyze_channels_for_img(inception, activation_model, img_tensor)

    layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
    analyze_filters(inception, layer_names)


if len(sys.argv) != 2:
    print("Usage: python AnalyzeModel.py <model_file>")
else:
    main(sys.argv[1])
