import tensorflow as tf 
import numpy as np
from data_prep import prep_data
from optimizer import Optimizer

"""

This class provides:
- Data generation (collocation point generation as well as bondary and initial condition training data preparation in prep_data)
- Scaling (for the nondimensionalized PINN)
- Optimizer instantiation for the training of the PINN

xbar = x/x_c

tbar = x_c/V (For a convection driven flow)

V 
"""


class PINNNondimensional: 


    def __init__(self, PINNModel, x_c, K, V):
        
        """
        Arguments: 
        - PINNModel: The PINN base class
        - x_c critical length scale of domain
        - K,V: lists of diffusion coefficients K and velocity constants V over which to train the network
        
        """
        self.K = K
        self.V = V
        self.x_c = x_c
        self.PINNModel = PINNModel # pass in the already constructed PINNModel

    def prep_data(self, Xrange = (0,1), Trange = (0,1),
                PeRange = (0.03,30), N_col = 80000, 
                N_ini = 400, N_bound = 400, 
                IC = lambda x: np.sin(np.pi*x),
                BC1 = lambda t: 0, BC2 = lambda t:0):


        """
        Prepares the non-dimensional data for training

        Arguments:
            - Xrange: Tuple indicating lower and upper bound of solution domain
            - Trange: Tuple indiciating lower and upper bound of solution domain
            - initialcondition: function taking as arguments x, outputting f(0,x)
            - boundarycondition1: function applied over Xrange[0] taking as arguments t. g1(t,Xrange[0])
            - boundarycondition2: function applied over Xrange[1] taking as arguments t. g2(t,Xrange[1])

        Outputs:
            x_train list containing:
            - tx_col: Input tx collocation matrix of shape (N_col, 2)
            - tx_init: Input tx initial condition matrix of shape (N_ini,2)
            - tx_bound: Input tx boundary condition matrix of shape (N_bound,2)

            y_train list containing:
            - c_col: output values for collocation matrix. Shape (N_col,1)
            - c_init: output values for initial condition matrix. Shape (N_ini,1)
            - c_bound: output values for boundary condition matrix. Shape (N_bound,1)

        """

        """
        Write x-training data
        """
        collocation_limits = np.array([Trange, Xrange, PeRange])
        sampling = LHS(xlimits=collocation_limits)

        t_col = sampling(N_col)

        tx_ini = sampling(N_ini)  
        tx_ini[:,0] = Trange[0]

        tx_bnd = sampling(N_bound)
        tx_bnd[:,1] = Xrange[1]*np.round(np.random.rand(N_bound,1))     


        """
        Write y-training data
        """

        c_col = np.zeros((N_col,1))

        c_ini = IC(tx_ini[:,1]).reshape(-1,1)
        
        lbound = tx_bnd[tx_bnd[:,1] == Xrange[0]]
        ubound = tx_bnd[tx_bnd[:,1] == Xrange[1]]
        tx_bnd = np.vstack((lbound,ubound))
        lbound[:,0] = BC1(lbound[:,0])  # Implement boundary conditions at upper and lower bounds
        ubound[:,0] = BC2(ubound[:,0])
        c_bnd = np.vstack((lbound,ubound))[:,0].reshape(-1,1)
        boundary = np.hstack((tx_bnd,c_bnd))
        np.random.shuffle(boundary)
        tx_bnd = boundary[:,0:2]
        c_bnd = boundary[:,2].reshape(-1,1)

        """
        Wrap-up training data to be sent to model
        """

        x_train = [t_col, tx_ini, tx_bnd]
        y_train = [c_col, c_ini, c_bnd]
        return x_train, y_train


    def scale_training_inputs(self, x_train):
        
        """
        Generate the scaled inputs for x_train, the list of values of K and V are iterated over in order to generate the range of training inputs

        Arguments:
            - x_train inputs: column [:,0] is time domain t, column [:,1] is position domain x
            - K: A list of diffusion coefficients over which the user wishes to train
            - V: A list of veliocities over which the user wishes to train

        Outputs:
            - scaled training data: column [:,0] is nondimensionalized time tbar,
            column [:,1] is nondimensionalized position xbar and column [:,2] is thePeclet number.

        """

        n = x_train.shape[0]
        [Kgrid,Vgrid] = np.meshgrid(self.K,self.V)
        KVgrid = np.hstack((Kgrid.flatten()[:,None], Vgrid.flatten()[:,None]))  # Develop list of combinations of K and V for training
        Pe = KVgrid[:,1]*self.x_c/KVgrid[:,0]  # Determine Pe = Vx/K

        n_scenarios = KVgrid.shape[0]

        tbar = np.zeros((n_scenarios,1))

        for i, Pec in enumerate(Pe):  # Find relevant time scales for diffusive and convective flows
            if Pec<=1:
                Pe[i] = 1
                tbar[i] = self.x_c**2/KVgrid[i,0]  # Diffusive flows x_c^2/K
            else:
                tbar[i] = self.x_c/KVgrid[i,1]  # Convective flows x_c/V_c

        #Scaling
        x_train[:,1]/=self.x_c
        n_scenarios = KVgrid.shape[0]
        scaled_inputs = np.zeros((n_scenarios*n,3))  # 3 input features, tbar, xbar, Pe, and n_scenarios*n observations

        for i in range(n_scenarios):

            Pe_i = np.repeat(Pe[i],n).reshape(-1,1)
            tbar_i = tbar[i]  # Find critical time scale for combination of K and V
            inputs_i = x_train
            inputs_i[:,0] /= tbar_i  # Scale time dimension by relevant time scale
            scaled_i = np.hstack((inputs_i, Pe_i))  # Stack inputs together
            scaled_inputs[i*n:(i+1)*n,:] = scaled_i  

        return scaled_inputs

        
    def optimizer_instantiate(self, IC, x_train, y_train, factr=1e5, m=50, maxls=50):
        """

        Instantiate the Scipy optimizer by passing the training inputs x_train and y_train

        Arguments:
            - x_train: x_col, x_ini, x_bound (produced by prep_data.py)
            - y_train: f_col, c_ini, c_bound (produced by prep_data.py)
        
        Outputs:
            - Optimizer object
        """
        #scale x_train data
        tx_col, tx_ini, tx_bnd = x_train

        tx_col_scaled = self.scale_training_inputs(tx_col)
        tx_ini_scaled = self.scale_training_inputs(tx_ini)
        tx_bnd_scaled = self.scale_training_inputs(tx_bnd)

        x_train = [tx_col_scaled,tx_ini_scaled,tx_bnd_scaled]

        #scale y_train data:
        
        f_col, c_ini, c_bnd = y_train
        
        f_col = np.zeros((tx_col_scaled.shape[0],1))
        c_ini = IC(tx_ini_scaled[:,1]).reshape(-1,1)
        c_bnd = np.zeros((tx_bnd_scaled.shape[0],1))

        y_train = [f_col, c_ini, c_bnd]

        optimizer = Optimizer(model=self.PINNModel, regularizer=1, x_train=x_train, y_train=y_train,maxiter= 200, factr=factr, m=m, maxls=maxls)

        return optimizer

    def predict(self, basemodel, K, V, x, t):

        """
        Use base model to predict flow field for given combination of K and V
        """
        model = basemodel
        n = len(x)

        #extract non-dimensional numbers
        x_c = self.x_c
        Pe = V*x_c/K

        if Pe<=1:
            Pe = 1
            t_c = x_c**2/K
        else:
            t_c = x_c/V        
        x_model_input = np.array(x/x_c)
        t_model_input = np.array(t/t_c) 
        Peclet_Input = np.repeat(Pe,n)
        inputs = np.stack([t_model_input,x_model_input,Peclet_Input],axis = -1)
        pred = model.predict(inputs)
        return(pred)

    

    
