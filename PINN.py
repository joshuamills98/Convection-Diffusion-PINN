import tensorflow as tf
from derivativelayer import DerivativeLayer

"""
Here we define the entire model for the Physics-Informed Neural Network
This will require the base Neural Network contained in:
basenetwork.py - this takes inputs x, t and outputs c
And the derivativelayer, contained in:
derivativelayer.py
"""


class PINNModel:

    """
    This will wrap the model up into its inputs and outputs  which are:
    Inputs:
    - Collocation inputs t and x over domain (txcol)
    - Inputs t and x over initial condition (txini)
    - Inputs t and x over boudnary condition (txbound)
    Outputs:
    - f_out: what we wish to minimize in order to regularize to the physics
    conditions (f_out = c_t +c_x-c_xx/Pe)
    - c_ini: initial conditions of the model (for data regularization)
    - c_bound: boundary conditions of the model (for data regularization)
    """
    def __init__(self, basemodel):

        self.basemodel = basemodel
        self.derivativelayer = DerivativeLayer(basemodel)

    def build(self):

        # Create inputs:
        tx_col_scaled = tf.keras.layers.Input(shape=(3, ))
        tx_ini_scaled = tf.keras.layers.Input(shape=(3, ))
        tx_bound_scaled = tf.keras.layers.Input(shape=(3, ))

        # f_out: (for conveciton diffusion equation we only need first
        # derivatives in x and t and second in x)
        _, dc_dt, dc_dx, _, d2c_dx2 = self.derivativelayer(tx_col_scaled)
        f_out = dc_dt+dc_dx-d2c_dx2/tx_col_scaled[:, 2]

        # c_ini: (for initial conditions we just evaluate base model
        # at corresponding x, t and Pe values)
        c_ini = self.basemodel(tx_ini_scaled)

        # c_bound
        c_bound = self.basemodel(tx_bound_scaled)
        f_out = tf.reshape(f_out, [-1, 1])
        return tf.keras.models.Model(
            inputs=[tx_col_scaled, tx_ini_scaled, tx_bound_scaled],
            outputs=[f_out, c_ini, c_bound]
        )
