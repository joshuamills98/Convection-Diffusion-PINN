import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Turn off TensorFlow warnings
import tensorflow as tf
import numpy as np
from basenetwork import set_flattened_weights
from derivativelayer import DerivativeLayer
from basenetwork import Network
from PINN import PINNModel
from optimizer import Optimizer
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from data_prep import prep_data
from PinnNonDimensional import PINNNondimensional
from plottingtools import plot_contour, ErrorAnalysisPlot
import pickle
import argparse

os.chdir("C:\\Users\\joshb\\Desktop\\Machine_Learning\\FYP Files\\MyFinalPINN\\Non-Dimensionalised Solver")

#Global Parameters for training
N_col = 10000 #Number of collocation points
N_bound_ini = 100 #total number of boundary and initial condition points
N_test_points = 1000
V = [0.1, 1, 2, 3]
K = [0.2, 0.9, 1.8, 3.2]
x_c = 1 #set critical length scale

# Set boundary and initial conditions
IC = lambda x: np.sin(np.pi*x)
bc1 = lambda t: 0
bc2 = lambda t: 0

#Model initialization
hidden_layers=[20]*9
basemodel = Network.basemodel(hidden_layers=hidden_layers)
PINN = PINNModel(basemodel).build()
PINNND = PINNNondimensional(PINN, x_c, K, V)
x_train, y_train = prep_data()
optimizer = PINNND.optimizer_instantiate(IC, x_train, y_train, factr=1e5, m=50, maxls=50)

parser = argparse.ArgumentParser(description= "Retrain selection")
parser.add_argument('retrain', type = str, default = 'No', help = "Train from scratch (Yes) or use pretrained weights (No)")
args = parser.parse_args()


weights = pickle.load( open( "weights-9.pkl", "rb" ))
flattened_weights = np.concatenate([np.array(w).flatten() for w in weights])
optimizer.set_weights(flattened_weights)



if __name__ == "__main__":

    if args.retrain == 'Yes':
        optimizer.fit(2500)

    if args.retrain == 'No':
        K_test =0.2
        V_test = 0.4
        xbounds = [0,1]
        tbounds = [0,1]
        plot_contour(basemodel, K_test, V_test, tbounds, xbounds, nt=1000, nx=50, levels=150, cmap='plasma', savename ='.\\Plots\\result.img', tslice = [0.05,0.3,0.6], save=False)     
