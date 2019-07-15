#Build CNN:
#Step 1: Convolution
#Step 2: Max MaxPooling
#Step 3: Flattening
#Step 4: Full Connection
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
#Input_shape order in documentation is the order for theano (3,64,64), not tensorflows!!!
#Tensorflow is opposite, (64,64,3)
#Convolution step takes image and creates feature map using a feature detector
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation= 'relu'))
#Compresses features map into smaller array by taking sum of each (2,2) spot in feature map
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#Takes all pool maps and creates long list of all features
#Second Convolutional layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, (3, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
#No input_dim since the flatten is the input, no kernel_init because it is pre-set in backend
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/OWNER/Documents/Projects/Summer2019/Tutorials/dl-udemy/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/OWNER/Documents/Projects/Summer2019/Tutorials/dl-udemy/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
