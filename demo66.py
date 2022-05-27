#將資料轉成矩陣one-hot vectors
from keras.utils import to_categorical

origs = 11#[4, 7, 16]
num_digits = 12
print(origs)
converted = to_categorical(origs, num_digits)
print(converted)