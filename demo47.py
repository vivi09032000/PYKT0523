import tensorflow as tf

@tf.function#graph
def add(p,q):
    return tf.math.add(p,q)

t1 = add([1,2,3],[4,5,6])
print(t1)
print(type(t1))