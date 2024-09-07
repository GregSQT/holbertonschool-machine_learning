#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
ubuntu@alexa-ml:~/tensorflow$ ./1-main.py 
Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)