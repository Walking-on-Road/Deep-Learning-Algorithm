#调整显卡资源分配
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
from sklearn.model_selection import train_test_split
backend.set_image_data_format('channels_last')

# 设定随机种子
seed = 7
np.random.seed(seed=seed)

# 导入数据
X_train , X_test, y_train, y_test = train_test_split(data,labels,test_size=0.4,random_state=100)
X_validation,X_test,y_validation,y_test = train_test_split(X_test,y_test,test_size=0.5,random_state=100)

# 格式化数据到0-1之前
X_train = np.array(X_train,dtype="float")
X_validation = np.array(X_validation,dtype="float")
X_test = np.array(X_test,dtype="float")
X_train = X_train / 255.0
X_validation = X_validation / 255.0
X_test = X_test/255.0

# one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

def create_model(epochs,classes):
    inputShape = (64, 64, 3)
    chanDim = -1
    model = Sequential()
    # CONV => RELU => POOL 
    model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL 
    model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL 
    model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # FC层
    model.add(Flatten())
    model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    # softmax 分类
    model.add(Dense(classes,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
    model.add(Activation("sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=3e-5), metrics=['accuracy'])
    return model

epochs = 100
classes = 2
model = create_model(epochs,classes)
hist = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=16, validation_data=(X_validation,y_validation),verbose=2)
scores = model.evaluate(x=X_test, y=y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))
