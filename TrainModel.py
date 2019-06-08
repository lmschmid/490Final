import sys
from keras import layers
from keras import models
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator


def get_generators(train_dir_name, val_dir_name, test_dir_name):
    # All images will be rescaled by 1./255
    train_data_gen = ImageDataGenerator(rescale=1./255)
    val_data_gen = ImageDataGenerator(rescale=1./255)
    test_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir_name,
        target_size=(299, 299),
        batch_size=32,
        class_mode="categorical")
    val_generator = val_data_gen.flow_from_directory(
        val_dir_name,
        target_size=(299, 299),
        batch_size=32,
        class_mode="categorical")
    test_generator = test_data_gen.flow_from_directory(
        test_dir_name,
        target_size=(299, 299),
        batch_size=1,
        class_mode=None
    )

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    return train_generator, val_generator, test_generator


def build_model():
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


def train_model(model, train_generator, val_generator, verbose=False):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_generator,
        validation_steps=50,
        verbose=verbose)


def test_model(model, test_generator):
    steps = test_generator.n // test_generator.batch_size

    predictions = model.predict_generator(
        test_generator,
        steps=steps,
        verbose=True)

    print(predictions)


def main(existing_model_path=None):
    train_dir_name = 'train/'
    val_dir_name = 'validation/'
    test_dir_name = 'test/'

    train_generator, val_generator, test_generator = get_generators(
        train_dir_name, val_dir_name, test_dir_name)

    if existing_model_path is not None:
        model = models.load_model(existing_model_path)
    else:
        model = build_model()
        train_model(model, train_generator, val_generator, verbose=True)
        model.save("./Model.h5")

    train_model(model, train_generator, val_generator, verbose=True)
    
    test_model(model, test_generator)


if len(sys.argv) > 1:
    main(sys.argv[1])
else:
    main()