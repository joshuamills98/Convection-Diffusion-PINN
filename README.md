# Convection-Diffusion-PINN
These files were developed as part of my Master's Thesis on *Physics-Informed Neural Networks (PINNs) within FLOW*.
Here, PINNs are used to solve the convection-diffusion PDE in the absence of a source given by:

![equation](https://latex.codecogs.com/gif.latex?u_%7Bt%7D%20%3D%20Vu_%7Bx%7D&plus;Du_%7Bxx%7D)


To my knowledge, PINNs have not yet been implemented to solve the convection-diffusion equation and no research performed on allowing the PINN to generalize over a range of system parameters (in this case D and V)
The goal is for the PINN to learn the solution 

![equation](https://latex.codecogs.com/gif.latex?NN%28t%2Cx%2CV%2CD%29%20%5Capprox%20u%28t%2Cx%2CV%2CD%29)

so that it can then be implemented within engineering software platforms and the solution does not need to rely on a more computationally taxing numerical method. 

The final neural network architecture had 4 input layers (t,x,V,D), 9 hidden layers, each with 20 neurons, and 1 output layer, u.
The network required ~90 minutes to train on an *Intel(R) Xeon(R) CPU @ 2.30GHz*.
The PINN is highly accurate in regions of high concentration and where the convection is well balanced with the diffusion and can solve the equation 8-9x quicker than a forward scheme.

![GitHub Logo](/result.png)

To implement this code, pretrained weights are provided so that the user does not need to retrain the PINN. Parse the argument 'No' to use these pretrained network weights and explore the solution by changing various parameters in the `if __name__ == "main"`
