import tensorflow as tf

"""
This file will compute the gradients in order to compute the f_loss and u_loss

"""


# Inherit from keras layers to use call method
class DerivativeLayer(tf.keras.layers.Layer):

    def __init__(self, model):

        self.model = model
        super(DerivativeLayer, self).__init__()

    def call(self, inputs):

        """
        Implement gradient tape method to compute derivatives

        inputs[:,0] is time
        inputs[:,1] is position

        """
        t = tf.convert_to_tensor(inputs[:, 0], dtype='float32')
        x = tf.convert_to_tensor(inputs[:, 1], dtype='float32')
        Pe = tf.convert_to_tensor(inputs[:, 2], dtype='float32')
        # tf.print(x)
        # tf.print(t)
        with tf.GradientTape(persistent=True) as gg:

            gg.watch(t)
            gg.watch(x)

            # Calculate output (i.e. concentration)
            c = self.model(tf.stack([t, x, Pe], axis=1))

            # Derive within tape so you can use tape after
            dc_dt = gg.gradient(c, t)
            dc_dx = gg.gradient(c, x)

        d2c_dt2 = gg.gradient(dc_dt, t)
        d2c_dx2 = gg.gradient(dc_dx, x)

        # Return all first and second derviatives
        return c, dc_dt, dc_dx, d2c_dt2, d2c_dx2
