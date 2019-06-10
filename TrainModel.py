import sys
import re
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import ResNet50, InceptionV3, VGG16, VGG19
from keras.preprocessing.image import ImageDataGenerator


def get_generators(train_dir_name, val_dir_name, test_dir_name, target_size):
    # All images will be rescaled by 1./255
    train_data_gen = ImageDataGenerator(rescale=1./255)
    val_data_gen = ImageDataGenerator(rescale=1./255)
    test_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir_name,
        color_mode="rgb",
        target_size=target_size,
        batch_size=32,
        class_mode="categorical")
    val_generator = val_data_gen.flow_from_directory(
        val_dir_name,
        color_mode="rgb",
        target_size=target_size,
        batch_size=32,
        class_mode="categorical")
    test_generator = test_data_gen.flow_from_directory(
        test_dir_name,
        color_mode="rgb",
        target_size=target_size,
        batch_size=1,
        class_mode=None,
        shuffle=False)

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    return train_generator, val_generator, test_generator


def build_inception_model():
    convBase = InceptionV3(
        weights='imagenet', 
        include_top=True, 
        input_shape=(299, 299, 3))
    convBase.trainable = False

    model = models.Sequential()
    model.add(convBase)
    model.add(layers.Dense(120, activation='softmax'))

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def build_custom_inception_model():
    convBase = InceptionV3(
        weights='imagenet', 
        include_top=False, 
        input_shape=(299, 299, 3))
    convBase.trainable = False

    model = models.Sequential()
    model.add(convBase)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(768, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(120, activation='softmax'))

    optimizer = optimizers.SGD(lr=0.001, momentum=0.09)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def build_custom_resnet_model():
    convBase = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3))
    convBase.trainable = False

    model = models.Sequential()
    model.add(convBase)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(768, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(120, activation='softmax'))

    optimizer = optimizers.SGD(lr=0.001, momentum=0.09)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def build_ensemble_model():
    input1 = layers.Input((299, 299, 3))
    input2 = layers.Input((299, 299, 3))
    input3 = layers.Input((299, 299, 3))
    input4 = layers.Input((299, 299, 3))

    inception_base = InceptionV3(
        weights='imagenet', 
        include_top=False)(input1)
    res_net_base = ResNet50(
        weights='imagenet',
        include_top=False)(input2)
    vgg_16_base = VGG16(
        weights='imagenet',
        include_top=False)(input3)
    vgg_19_base = VGG19(
        weights='imagenet',
        include_top=False)(input4)

    inception_base.trainable, res_net_base.trainable, vgg_16_base.trainable, \
        vgg_19_base.trainable = False, False, False, False

    flatten1 = layers.Flatten()(inception_base)
    flatten2 = layers.Flatten()(res_net_base)
    flatten3 = layers.Flatten()(vgg_16_base)
    flatten4 = layers.Flatten()(vgg_19_base)

    flatten1.trainable, flatten2.trainable, flatten3.trainable, \
        flatten4.trainable = False, False, False, False

    last_layers = layers.concatenate([flatten1, flatten2])
    last_layers = layers.Dense(128, activation='relu')(last_layers)
    last_layers = layers.Dropout(0.5)(last_layers)
    last_layers = layers.Dense(120, activation='softmax')(last_layers)

    ensemble_model = models.Model(
        inputs=[input1, input2], 
        outputs=last_layers)
    optimizer = optimizers.SGD(lr=0.001, momentum=0.09)
    ensemble_model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return ensemble_model


def train_model(model, train_gen, val_gen, epochs=30, verbose=False):
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=50,
        verbose=verbose)

    return history


def classify_images(model, label_map, test_generator, verbose=False):
    steps = test_generator.n

    predictions = model.predict_generator(
        test_generator,
        steps=steps,
        verbose=verbose)

    labels = sorted(list(label_map.keys()))
    
    with open("predictions.csv", 'w') as pred_file:
        pred_file.write('id,{}\n'.format(",".join(labels)))
        for index, prediction in enumerate(predictions):
            id = re.split("[./]", test_generator.filenames[index])[-2]
            prediction_list = prediction.tolist()
            confidence_vals = ",".join(map(str, prediction_list))
            pred_file.write("{},{}\n".format(id, confidence_vals))

            if verbose:
                max_val = labels[prediction_list.index(max(prediction_list))]
                print("Image '{}' classified as a {}".format(id, max_val))


def ensemble_input_generator(generator):
    for image, label in generator:
        yield [image, image, image, image], label


def main(existing_model_path=None, model_type=None):
    train_dir_name = 'train/'
    val_dir_name = 'validation/'
    test_dir_name = 'test/'

    if existing_model_path is not None:
        model = models.load_model(existing_model_path)

    elif model_type == "InceptionBase":
        train_generator, val_generator, test_generator = get_generators(
            train_dir_name, val_dir_name, test_dir_name, (299, 299))
        model = build_inception_model()
        train_model(model, train_generator, val_generator, 
            epochs=30, verbose=True)
        model.save("./InceptionBase.h5")

    elif model_type == "Inception":
        train_generator, val_generator, test_generator = get_generators(
            train_dir_name, val_dir_name, test_dir_name, (299, 299))
        model = build_custom_inception_model()
        train_model(model, train_generator, val_generator, 
            epochs=50, verbose=True)
        model.save("./Inception.h5")

    elif model_type == "ResNet":
        train_generator, val_generator, test_generator = get_generators(
            train_dir_name, val_dir_name, test_dir_name, (224, 224))
        model = build_custom_resnet_model()
        train_model(model, train_generator, val_generator, 
            epochs=50, verbose=True)
        model.save("./ResNet.h5")

    elif model_type == "Ensemble":
        train_generator, val_generator, test_generator = get_generators(
            train_dir_name, val_dir_name, test_dir_name, (299, 299))
        model = build_ensemble_model()
        ensemble_train_generator = ensemble_input_generator(train_generator)
        ensemble_val_generator = ensemble_input_generator(train_generator)
        train_model(model, ensemble_train_generator, ensemble_val_generator, 
            epochs=100, verbose=True)
        model.save("./Ensemble.h5")
        
    else:
        print("Error: {} is not a valid type".format(model_type))

    classify_images(
        model, 
        train_generator.class_indices, 
        test_generator, 
        verbose=True)


if "--type" in sys.argv:
    main(model_type=sys.argv[sys.argv.index("--type") + 1], existing_model_path=None)
elif "--model" in sys.argv:
    main(model_type=None, existing_model_path=sys.argv[sys.argv.index("--model") + 1])
else:
    print("Usage: python3 TrainModel.py [--type <type (InceptionBase, Inception, ResNet, Ensemble)> <output_model_name>] [--model <model_file_path>]")