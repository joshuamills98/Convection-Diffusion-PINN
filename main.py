import os
import numpy as np
from basenetwork import Network
from PINN import PINNModel
from PinnNonDimensional import PINNNondimensional
from plottingtools import plot_contour
import pickle
import argparse
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Turn off TensorFlow warnings

# Global Parameters for training
N_col = 10000  # Number of collocation points
N_bound_ini = 100  # total number of boundary and initial condition points


# Train over the following velocity constants and diffusivity coefficients
V = [0.1, 1, 2, 3]
K = [0.2, 0.9, 1.8, 3.2]
x_c = 1  # Set critical length scale

# Set boundary and initial conditions


def IC(x):
    return np.sin(np.pi*x)


# def BC1(t):
#     return 0

# def BC2(t):
#     return 0

# Model initialization
hidden_layers = [20]*9
basemodel = Network.basemodel(hidden_layers=hidden_layers)
PINN = PINNModel(basemodel).build()
PINNND = PINNNondimensional(PINN, x_c, K, V)
x_train, y_train = PINNND.prep_data()
optimizer = PINNND.optimizer_instantiate(IC,
                                         x_train, y_train,
                                         factr=1e5,
                                         m=50,
                                         maxls=50)

parser = argparse.ArgumentParser(description="Retrain selection")
parser.add_argument(
    'retrain',
    type=str,
    default='No',
    help="Train from scratch (Yes) or use pretrained weights (No)")

args = parser.parse_args()

if __name__ == "__main__":

    if args.retrain == 'Yes':
        optimizer.fit(2500)
        with open('NNWeights9x20-{}.pickle'
                  .format(time.strftime("%H-%M-%S")), 'wb') as handle:
            pickle.dump(basemodel.weights, handle)

    if args.retrain == 'No':
        weights = pickle.load(open("weights-9.pkl", "rb"))
        flattened_weights = np.concatenate(
            [np.array(w).flatten() for w in weights])
        optimizer.set_weights(flattened_weights)

        K_test = 0.2
        V_test = 0.4
        xbounds = [0, 1]
        tbounds = [0, 1]
        plot_contour(basemodel,
                     K_test, V_test,
                     tbounds, xbounds,
                     nt=1000, nx=50,
                     levels=150,
                     cmap='plasma',
                     savename='.\\Plots\\result.img',
                     tslice=[0.05, 0.3, 0.6], save=False)
