from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialize CNN model
classifier = Sequential()

# Layer-1 Convolution
classifier.add(Conv2D(96,(11,11),strides=(4,4), input_shape = (150,220,3), activation = 'relu'))

# Layer-2 Pooling
classifier.add(MaxPooling2D(pool_size = (3,3)))

# Layer-3 Convolution
classifier.add(ZeroPadding2D(padding=(2, 2)))
classifier.add(Conv2D(256,(5,5),strides=(2,2), activation = 'relu'))

# Layer-4 Pooling
classifier.add(MaxPooling2D(pool_size = (3,3)))

# Layer-5 Convolution
classifier.add(ZeroPadding2D(padding=(1, 1)))
classifier.add(Conv2D(384,(3,3),strides=(1,1), activation = 'relu'))

# Layer-6 Convolution
classifier.add(ZeroPadding2D(padding=(1, 1)))
classifier.add(Conv2D(384,(3,3),strides=(1,1), activation = 'relu'))

# Layer-7 Convolution
classifier.add(ZeroPadding2D(padding=(1, 1)))
classifier.add(Conv2D(256,(3,3),strides=(1,1), activation = 'relu'))

# Layer-8 Pooling
# classifier.add(MaxPooling2D(pool_size = (3,3)))

# Layer-9 Flattening
classifier.add(Flatten())

# Layer-10 Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset', target_size = (150, 220), batch_size = 32, class_mode = 'binary')

test_set = test_datagen.flow_from_directory('sample_Signature', target_size = (150, 220), batch_size = 32, class_mode = 'binary')

from IPython.display import display
from PIL import Image

classifier.fit_generator(training_set, steps_per_epoch = 1712, epochs = 20, validation_data = test_set, validation_steps = 800)

classifier.save('offline-sign-CNN-01.h5')
del classifier

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('G1.png', target_size = (150, 220))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] >= 0.5:
    prediction = 'Genuine'
else:
    prediction = 'forged'
print(prediction)

test_image = image.load_img('G2.png', target_size = (150, 220))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] >= 0.5:
    prediction = 'Genuine'
else:
    prediction = 'forged'
print(prediction)

test_image = image.load_img('F1.png', target_size = (150, 220))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] >= 0.5:
    prediction = 'Genuine'
else:
    prediction = 'forged'
print(prediction)

test_image = image.load_img('G112.png', target_size = (150, 220))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] >= 0.5:
    prediction = 'Genuine'
else:
    prediction = 'forged'
print(prediction)

test_image = image.load_img('S135.png', target_size = (150, 220))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] >= 0.5:
    prediction = 'Genuine'
else:
    prediction = 'forged'
print(prediction)
