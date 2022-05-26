import tensorflow as tf

tf.compat.v1.disable_eager_execution()
t1 = tf.constant("hello")
print(t1)

session1 = tf.compat.v1.Session()
print(session1.run(t1))
session1.close()