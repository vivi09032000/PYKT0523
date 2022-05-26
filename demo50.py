import tensorflow as tf

@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


c1 = tf.constant([[3.0, 4.0, 5.0],
                  [6.0, 6.0, 6.0],
                  [2.5, 4.1, 4.0]])

print(computeArea(c1).numpy())