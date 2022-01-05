# <center> Using _Physics Informed Neural Network_ to solve the Convection-Diffusion Equation </center>

## <center> **Overview** </center>
This program was developed as part of my Master's Thesis on *The Implementation of Physics-Informed Neural Networks (PINNs) within FLOW*. [Flow](https://www.theengineeringcompany.com/) is a virtual engineering design platform developed by the Engineering Company. 

Within virtual engineering design, simulations are performed with numerical solvers and are often computationally intensive, preventing the user from rapidly testing designs. In this project I sought to implement PINNs to solve different partial and ordinary differential equations and compare the results with traditional numerical solvers. The goal was to demonstrate speed increases through the implementation of PINNs and to also allow the PINNs to generalize over a broad range of solutions.

In this particular example I apply the PINN to a common partial-differential equaition that governs fluid flow driven by convection and diffusion. Through the applciation of PINNs I was able to demonstrate a significant speed increase without a substantial drop in accuracy.

NOTE: This readme is fairly brief - for a more detailed explanation of PINNs, I would recommend reviewing [Maziar Riassi's explanation](https://maziarraissi.github.io/PINNs/). 

## <center> **Methodology** </center>

Here, PINNs are used to solve the convection-diffusion PDE in the absence of a source given by:

![equation](https://latex.codecogs.com/gif.latex?\theta_%7Bt%7D%20%3D%20V\theta_%7Bx%7D&plus;D\theta_%7Bxx%7D)

To my knowledge, PINNs have not yet been implemented to solve the convection-diffusion equation and no research performed on allowing the PINN to generalize over a range of system parameters (in this case D and V). The goal is for the PINN to learn the solution the above PDE so that:

![equation](https://latex.codecogs.com/gif.latex?NN%28t%2Cx%2CV%2CD%29%20%5Capprox%20\theta%28t%2Cx%2CV%2CD%29)

The PINN that has been developed nondimensionalizes the equation to give:

![equation](
https://latex.codecogs.com/gif.latex?%5Cbar%7B%5Ctheta%7D_%7B%5Cbar%7Bt%7D%7D%20&plus;%20%5Cbar%7B%5Ctheta%7D_%7B%5Cbar%7Bx%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BPe%7D%5Cbar%7B%5Ctheta%7D_%7B%5Cbar%7Bx%7D%5Cbar%7Bx%7D%7D)

_Pe_ is the Peclet number and describes the balance between convective and diffusive forces within the fluid flow. (The derivation relies on assumptions about the flow, they are detailed in the report).

To train the PINN to solve the PDE, the architecture below was adopted:
<p align="center">
  <img width=50% height=50% src="/plots/ConvDiffusionNDPINN.png">
</p>

The PINN is governed by the following idea: for the Neural Network to be an accurate solution to the underlying PDE, it must satisfy both the boundary conditions of the PDE as well as the underlying PDE itself (given above)
In the above diagram, _MSE_u_ is the loss associated with the PINN applied to the boundary conditions. The PINN is then passed through a set of differential operators which are used to determine the _MSE_f_. _MSE_f_ is the loss associated with the PINN when used in the PDE, this is called the _Residual Network_. The combination of both these loss terms is used with an optimizer to perform backward propogation and optimize the neural network weights.

The network is trained over a series of boundary points to determine _MSE_u_ and collocation points (those placed in the domain of x,t and Pe) to determine _MSE_f_.
For my training I used over 90,000 collocation, dispersed using latin hypercube sampling below.
<p align="center">
  <img width=70% height=70% src="/plots/DataDistribution.png">
</p>

The final neural network architecture had 4 input layers (t,x,V,D), 9 hidden layers, each with 20 neurons, and 1 output layer, u. The PINN was trained over
The network required ~90 minutes to train on an *Intel(R) Xeon(R) CPU @ 2.30GHz*.

## <center> **Results** </center>
The PINN is highly accurate in regions of high concentration and where the convection is well balanced with the diffusion and can solve the equation **8-9x quicker** than a forward Euler scheme. 

Attached shows the solution for *D=0.2 m^2 s^-1* and *V=0.4 m s^-1*

<p align="center">
  <img width="1000" height="500" src="/plots/result.png">
</p>
The PINN is most accurate in describing flows with a reasonable balance between convective and diffucsive forces. In regions of high convectivity, the PINN's accuracy drops significantly. This is demonstrated in the accuracy plot below:

<p align="center">
  <img width="600" height="600" src="/plots/Error2.png">
</p>

There is a clear trade-off between generalizability and accuracy - further work should seek to combine a hybrid approach where multiple PINNs are used to capture the different regions of the solution (convective/diffusive region).

## <center> Implementation within FLOW </center>



# Files

* **basenetwork.py -** Underlying neural network structure 
* **derivativelayer.py -** Residual network operations
* **PINN.py -** The entire construction of the PINN is wrapped up here
* **optimizer.py -** Creation of the optimizer object for training of the PINN
* **plottingtools.py -** Tools for plotting the error and results of the PINN
* **NNWights-9pickle -** Pre-trained weights with 9 hidden layers and 20 neurons per layer to use for exploration of the solution
* **PinnNonDimensional.py -** The nondimensionalization takes place here as well as all the data preparation required for training the PINN

To implement this code, pretrained weights are provided so that the user does not need to retrain the PINN. Parse the argument 'No' to use these pretrained network weights and explore the solution by changing various parameters in the `if __name__ == "main"`


