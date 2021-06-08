# Convection-Diffusion-PINN
These files were developed as part of my Master's Thesis on Physics-Informed Neural Networks (PINNs) within FLOW.
Here, PINNs are used to solve the convection-diffusion PDE in the absence of a source given by:

![equation](https://latex.codecogs.com/gif.latex?u_%7Bt%7D%20%3D%20Vu_%7Bx%7D&plus;Du_%7Bxx%7D)


To my knowledge, PINNs have not yet been implemented to solve the convection-diffusion equation and no research performed on allowing the PINN to generalize over a range of system parameters (in this case D and V)
The goal is for the PINN to learn the solution https://latex.codecogs.com/gif.latex?NN%28t%2Cx%2CV%2CD%29%20%5Capprox%20u%28t%2Cx%2CV%2CD%29 so that it can then be implemented within engineering software platforms
and the solution does not need to rely on a more computationally taxing numerical method. 
The PINN is highly accurate in regions where the convection is well balanced with the diffusion and can solve the equation 8-9x quicker than a forward scheme.

![alt text](https://github.com/[joshuamills98]/[Convection-Diffusion-PINN]/blob/[master]/result.png?raw=true)

