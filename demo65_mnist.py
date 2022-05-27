from matplotlib import pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#計算出維度
print(train_images.shape, test_images.shape)
print(len(train_labels), len(test_labels))

#劃出mnist的圖，標題是對應的label
def plotImage(index):
    plt.title("the image marked as:%d" % train_labels[index])
    plt.imshow(train_images[index], cmap='binary')
    plt.show()


def plotTestImage(index):
    plt.title("test image marked as:%d" % test_labels[index])
    plt.imshow(test_images[index], cmap='binary')
    plt.show()


plotImage(200)
plotTestImage(200)