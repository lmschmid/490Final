from keras import layers
from keras import models
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator


trainDirName = 'train/'
valDirName = 'validation/'
testDirName = 'test/'

# All images will be rescaled by 1./255
trainDatagen = ImageDataGenerator(rescale=1./255)
valDatagen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDatagen.flow_from_directory(
    trainDirName,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical")
valGenerator = trainDatagen.flow_from_directory(
    valDirName,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical")

for data_batch, labels_batch in trainGenerator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


convBase = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299, 299, 3))
convBase.trainable = False

model = models.Sequential()
model.add(convBase)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(120, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=valGenerator,
    validation_steps=50)
