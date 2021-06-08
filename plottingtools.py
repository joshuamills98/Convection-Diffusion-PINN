import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
plt.style.use('bmh')
plt.rcParams.update({'font.size': 14})

def find_nearest(array, value):

    """
    Helper function for plotting
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def numerical_solver(x, t, K, V):

    """
    Solves the convection diffusion equation numerically using a basic forward scheme.

    Arguments:
        x: spatial domain over which to solve
        t: temporal domain over which to solve 
        V: Convective coefficient
        K: Diffusive coefficient
    
    Outputs:
        Solutiontime: Time to solve
        Tmat: Matrix of solved values
    """
    nx = len(x)
    nt = len(t)
    T = np.zeros((nx,nt))
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    start = time.time()
    T[:,0] = np.sin(np.pi*x)

    for ti in range(1,nt-1):
        for xi in range(1,nx-1): # domain
            T[xi,ti] = (1-2*K*dt/dx**2)*T[xi,ti-1]+(K*dt/dx**2+V*dt/(2*dx))*T[xi-1,ti-1]+(K*dt/dx**2-V*dt/(2*dx))*T[xi+1,ti-1]
    
    end = time.time()
    solution_time = end-start

    return T, solution_time

def plot_contour(
                basemodel, K, V, Tmax, tbounds=[0,1],
                xbounds=[0,1], tslice=[0, 0.3,0.6], 
                nt=1000, nx=40, levels=30, 
                cmap='plasma', savename='.\\Plots\\result.img', 
                save=False):
    
    """
    Plotting function that will take care of scaing and pre-processing.

    Arguments:
        basemodel: base Neural Network that accepts scaled x,t and Pe
        K: Diffusion Constant
        V: Velocity Constant
        N_test_points: total number of domain poinnts over x and t will be N_test_points^2
        tbounds: list or tuple of lower and upper bounds of t
        xbounds: list or tuple of lower and upper bounds of x

    Outputs:
        Plot of Flow Pattern

    """
    x_c = xbounds[1]
    Pe = V*x_c/K  # Determine Peclet number 

    if Pe<=1:  # Scale inputs based on Peclet number
        Pe = 1
        t_c = x_c**2/K
    else:
        t_c = x_c/V        
    

    tnum = np.linspace(tbounds[0],tbounds[1],nt)
    xnum = np.linspace(xbounds[0],xbounds[1],nx)
    t, x = np.meshgrid(tnum, xnum)

    Peclet_Input = np.repeat(Pe,nt*nx)

    txPe = np.stack([t.flatten()/t_c, x.flatten()/x_c, Peclet_Input], axis=-1)
    u = basemodel.predict(txPe, batch_size=nx*nt)
    u = u.reshape(t.shape)

    fig = plt.figure(figsize=(15,11))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    cs1 = plt.contourf(t,x,u,cmap = cmap,levels = levels)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    cbar = plt.colorbar(cs1)
    cbar.set_label(r'$\theta(t,x)$')
    
    t_cross_sections = tslice  # Plot cross-sections to compare with numerical solution
    
    for cross_section in t_cross_sections:  # Demonstrate where cross-sections are being taken on main contour plot
        plt.axvline(cross_section, c = 'w', linestyle = '--')

    """
    Numerical Solution
    """

    T, solution_time = numerical_solver(xnum, tnum, K,V)

    """
    Comparing Cross Sections
    
    """

    idx = []
    tnew = []
    for t in t_cross_sections:
        index, val = find_nearest(tnum, t)
        tnew.append(val)
        idx.append(index)
        
    for i, t_cs in enumerate(tnew):

        plt.subplot(gs[1, i])
        txPe = np.stack([np.full(xnum.shape, t_cs/t_c), xnum/x_c, np.full(xnum.shape, Pe)], axis=-1)
        u = basemodel.predict(txPe, batch_size=nx*nt)

        plt.plot(xnum, u, label = 'PINN Prediction')
        plt.plot(xnum,T[:,idx[i]], color = 'red', linestyle='dashed', label = 'Numerical Solution')
        plt.title(r'$t = {:.2f}$'.format(t_cs))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\theta(t,x)$')
        if i==0:
            plt.legend(fontsize = 'xx-small', loc = 'lower center')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(savename, dpi = 300, transparent=True)
    plt.show()
    
    return solution_time 


def ErrorAnalysisPlot(  
                    basemodel, nt=1000,nx=40, n_scenarios=4, 
                    KRange = [0.001,4], VRange = [0.001, 4], 
                    tbounds = [0,1], xbounds = [0,1], levels=30,
                    training_points = ([0.1,1,2,3], [0.2,0.9,1.8,3.2])):
                    
    """

    Plot the error between numeric solution and PINN over a specified range of K and V values
    For the numerical scheme to be stable nt>>nx

    """
    """
    Set up K and V grid over which different solutions will be compared

    """
    x_c = xbounds[1]
    K = np.linspace(KRange[0],KRange[1],n_scenarios)
    V = np.linspace(VRange[0],VRange[1],n_scenarios)

    [Kgrid,Vgrid] = np.meshgrid(K,V)
    KVgrid = np.hstack((Kgrid.flatten()[:,None], Vgrid.flatten()[:,None]))
    Pe = KVgrid[:,1]*x_c/KVgrid[:,0]

    tnum = np.linspace(tbounds[0],tbounds[1],nt) #set up domain
    xnum = np.linspace(xbounds[0],xbounds[1],nx)
    dx = xnum[1]-xnum[0]
    dt = tnum[1]-tnum[0]

    Error = []

    for i in range(n_scenarios**2):  # List over all the K and V and determine the error between numerical scheme and PINN for each combination 
 
        K = KVgrid[i,0]
        V = KVgrid[i,1]
        Pe = V*x_c/K

        assert(dx < 2*K/V)  # Assert stability conditions for forward numerical solver
        assert(dt < dx**2/(2*K))

        T, solution_time = numerical_solver(xnum, tnum, K,V)  # Find numerical solution

        if Pe<=1:  # Scale inputs based on Peclet number
            Pe = 1
            t_c = x_c**2/K
        else:
            t_c = x_c/V
        
        Peclet_Input = np.repeat(Pe,nt*nx)
        t, x = np.meshgrid(tnum/t_c, xnum/x_c)
        txPe = np.stack([t.flatten(), x.flatten(), Peclet_Input], axis=-1)
        u = basemodel.predict(txPe, batch_size=nt*nx)  # Find PINN solution

        u = u.reshape(T.shape)  # Determine error between numerical scheme and PINN
        Error.append(np.log(np.abs((np.subtract(u, T))).mean()))  

    Error = np.array(Error)
    Error = Error.reshape((n_scenarios,n_scenarios))
    
    fig = plt.figure(figsize=(10,8))  # Plot error map
    cs1 = plt.contourf(Kgrid,Vgrid,Error, levels = levels)
    x1,x2 = np.meshgrid(training_points[0],training_points[1])
    grid = np.hstack((x1.flatten()[:,None], x2.flatten()[:,None]))
    plt.plot(grid[:,0],grid[:,1],'rx',alpha=0.5, label = 'Training Points')
    plt.plot()
    plt.xlabel('Diffusion Coefficient, K (m^2/s)')
    plt.ylabel('Velocity of Fluid, V (m/s)')
    plt.legend()
    cbar = plt.colorbar(cs1)
    cbar.set_label('Error')
    
    return Error










    