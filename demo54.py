#客製化梯度
import tensorflow as tf

time = tf.Variable(5.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 *time**2
    speed = inner_tape.gradient(position,time)
    print("speed=",speed)
accelerator = outer_tape.gradient(speed,time)
print("accelerator=",accelerator)