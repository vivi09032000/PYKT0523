#深度學習類神經
import numpy
from keras.layers import Dense
from keras.models import Sequential
import keras
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs", histogram_freq=1)
dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]
print(inputList.shape, resultList.shape)


def createModel():
    #global model
    m = Sequential()
    layers = [Dense(20, input_dim=8, activation='relu'),
              Dense(8, activation='relu'),
              Dense(1, activation='sigmoid')]
    for l in layers:
        m.add(l)
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m

model = createModel()

#validation_split=0.25 training會拿1/4出來做validate
#val_accuracy是已經train過一次的data
model.fit(inputList, resultList, epochs=100, batch_size=20,validation_split=0.25,callbacks=[tensorboard])
