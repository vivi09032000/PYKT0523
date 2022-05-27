#深度學習類神經
import numpy
from keras.layers import Dense
from keras.models import Sequential
import keras

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

model.fit(inputList, resultList, epochs=100, batch_size=20)

scores = model.evaluate(inputList, resultList)

metrics = model.metrics_names
for s, m in zip(scores, metrics):
    print(f"{m} score={s}")
# print("scores=", scores)
# 手動建一個model目錄
#把train好的model儲存起來
keras.models.save_model(model, "model/demo51")

model2 = createModel()
scores2 = model2.evaluate(inputList, resultList)
metrics2 = model2.metrics_names
for s, m in zip(scores2, metrics2):
    print(f"[model2]:{m} score={s}")

#叫出model使用，結果會跟model1是一樣的
model3 = keras.models.load_model("model/demo51")
scores3 = model3.evaluate(inputList, resultList)
metrics3 = model3.metrics_names
for s, m in zip(scores3, metrics3):
    print(f"[model3]:{m} score={s}")