import scipy
from scipy.stats import linregress
from scipy.optimize import fsolve
import numpy as np
from math import ceil
from numpy import log10
import matplotlib.pyplot as plt
from typing import Callable


# Different utilities --------------------------------------------
def signif(x, p): 
    """Stack Overflow magic. Writes x with p number of significant digits."""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def namestr(obj, namespace):
    """Stack Overflow magic. Returns name of array/variable."""
    return [name for name in namespace if namespace[name] is obj][0]

def K_norm(M:np.array):
    """Calculates the K-norm defined in my master's thesis of a vector M or of each column of a matrix M."""
    if len(M.shape)==1:
        return (np.abs(M[0])/2 + np.abs(M[-1])/2 + np.sum(np.abs(M[1:-1])) )/(M.shape[0]-1)
    elif len(M.shape)==2:
        return np.array([ (M[0,i]/2 + M[-1,i]/2 + np.sum(M[1:-1,i])) 
                           for i in range(M.shape[1])])/(M.shape[0]-1)
    else:
        raise Exception(f"K_norm only supports vectors or matrices. Input has shape {M.shape}")

def _calc_rel_error(numerical:np.array, analytic:np.array, errortype:str)-> np.float64:
    """Calculates the relative difference between numerical and analytic with the norm given by errortype."""
    if errortype=="infty":
        return np.max(np.abs(numerical-analytic))/np.max(np.abs(analytic))
    elif errortype=="K":
        return np.max(K_norm(numerical-analytic))/np.max(K_norm(analytic))
    else:
        raise Exception("Only errortypes 'infty' and 'K' are supported.")

def roll(a, shift):
    """np.roll but instead of having a "circular" array, elements that roll beyond the last position disappear and the missing value stays as it did before. When doing difference methods, corresponds to homogenous Neumann boundary conditions."""
    ret = np.roll(a,shift)
    if shift>0:
        ret[0]=a[0]
    elif shift<0:
        ret[-1]=a[-1]
    return ret


# Plotting functions ----------------------------------------------
def plot_different_times(numeric_sol:np.array, analytic_fnc:Callable, setup_dict:dict, 
                         title=None, filename=None, ylabel="y",**subplots_kwargs)->None:
    """Plots the numeric_sol vs analytic_fnc at 9 equidistant timepoints.

    Args:
        numeric_sol (np.array): Numerical solution as (N_x+1)x(N_t+1)-matrix
        analytic_fnc (Callable: Analytic solution as a function of (x,t)
        setup_dict (dict): Dictionary as produced by "setup" or "setup_FP_test"
        title (str, optional): Title for plots. Defaults to None.
        filename (str, optional): Saves the plots under "/Images/<filename>.pdf". Defaults to None.
        ylabel (str, optional): y-label of plots. Defaults to "y".
    """
    t=setup_dict["t"]
    x=setup_dict["x"]
    
    fig,axs = plt.subplots(3,3, figsize=(7,6), sharex=True, **subplots_kwargs)
    t_indexes =np.int64(np.round(np.linspace(0,len(t)-1,9)))
    for i in range(3):
        for j in range(3):
            t_index = t_indexes[i*3+j]
            axs[i,j].set(title=f"Time {np.round(t[t_index],2)}")
            if type(analytic_fnc)!=type(None):
                axs[i,j].plot(x, analytic_fnc(x,np.ones(x.shape)*t[t_index]), 
                              label="Analytic", linestyle="-", color="orange")
            if type(numeric_sol)!=type(None):
                axs[i,j].plot(x, numeric_sol[:,t_index], label="Numeric",
                              linestyle="", marker="o", markersize=1)
            axs[i,j].grid()

    axs[0,0].legend(fontsize=7.5)
    fig.supxlabel('x')
    fig.supylabel(ylabel)
    if title!=None:
        plt.suptitle(title)
    fig.tight_layout()
    if filename:
        plt.savefig("Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()

def plot_total_mass(h:np.float64, t:np.float64, M:np.array, filename=None, **kwargs)->None:
    """Plots the total mass (K-norm on the columns of M) over time.

    Args:
        h (np.float64): Spatial stepsize
        t (np.float64): Temporal stepsize
        M (np.array): Mass density as (N_x+1)x(N_t+1)-matrix.
        filename (str, optional): Saves the plots under "/Images/<filename>.pdf". Defaults to None.
    """
    total_mass = np.array([ (M[0,i]/2 + M[-1,i]/2 + np.sum(M[1:-1,i])) 
                           for i in range(M.shape[1])])*h
    
    min_mass_diff = abs(1-np.min(total_mass))
    max_mass_diff = abs(np.max(total_mass)-1) 
    interval =max(max(min_mass_diff,max_mass_diff), 0.01)
    
    plt.plot(t, total_mass, linestyle='', marker='o', markersize=1, label="Total mass $\|M\|_{\mathcal{K}}$ over time")
    plt.ylim(1-1.1*interval,1+1.1*interval)
    plt.xlabel("time $t$")
    plt.ylabel("$\|M\|_{\mathcal{K}}$")
    plt.grid()
    plt.legend()
    if filename:
        plt.savefig("Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()

def _MFG_solver_plot_convergence_of_fixed_point_iterations(U_change:np.array, M_change:np.array,a:np.float64, 
                                                           U_error=None, M_error=None, filename =None)->None:
    """Plots evolution of the fixed-point iteration solving the dicretized MFG system.

    Args:
        U_change (np.array): List with relative change in U for the different iteration-steps.
        M_change (np.array): List with relative change in M for the different iteration-steps.
        a (np.float64): Memory as given in the fixed-point iteration.
        U_error (np.array, optional):  List with relative error in U for the different iteration-steps. Defaults to None.
        M_error (np.array, optional): List with relative error in M for the different iteration-steps. Defaults to None.
        filename (str, optional): Saves the plots under "/Images/<filename>.pdf". Defaults to None.
    """
    markersize=3.5
    if type(U_error)!=None:
        fig, (ax1,ax2)=plt.subplots(1,2)
        ax2.plot(U_error,label="Relative $L^\infty$ error in U",
                    marker="o", markersize=markersize)
        ax2.plot(M_error, label="Relative $\mathcal{K}$ error in M",
                    marker="o", markersize=markersize)
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid()
    else:
        fig, ax1 = plt.subplots()
    fig.suptitle("Evolution of fixed point iteration, $\mathcal{M}^{j+1}=$"+f"{signif(1-a,2)}" + "$\chi(\mathcal{M}^j)+"+f"{signif(a,2)}"+"\mathcal{M}^j$")
    ax1.plot(U_change,label="Relative $L^\infty$ change in U",
                marker="o", markersize=markersize)
    ax1.plot(M_change, label="Relative $\mathcal{K}$ change in M",
                marker="o", markersize=markersize)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid()
    
    if filename:
        plt.savefig("Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()

def loglog(solver, N_xs, test_object, g, first_reg_node=0, 
           last_reg_node=-1, verbose=1, title=None, filename=None):
    """Applies "solver" with numerical Hamiltonian "g" to the existing "test_object" for each of the N_x's in the list "N_xs" (note that h=Delta t). Then a loglog plot is created which also includes a line representing the rounded numerical convergence rate, which is calculated from N_xs[first_reg_node:last_reg_node].

    Args:
        solver: A solver for either the HJB- or the FP-equation. Usually "HJ_implicit" or "FP_implicit"
        N_xs: List containing the number of spatial steps to consider. E.g. [10,20,40,80,160,320]
        test_object: A test-object which inherits from "Test_HJ" or "Test_FP" in "test_problems.py".
        g: Numerical Hamiltonian. "godunov" is implemented in this code.
        first_reg_node (optional: Start of interval in "N_xs" to use to calculate numerical convergence rate. Defaults to 0.
        last_reg_node (optional): End of interval in "N_xs" to use to calculate numerical convergence rate.. Defaults to -1.
        verbose (optional): If 1, prints the relative error for each element in "N_xs". If 0, it won't. Defaults to 1.
        title (optional): . Gives the plot a title. Defaults to None.
        filename (optional): Saves the plots under "/Images/<filename>.pdf". Defaults to None.
    """
    
    N_xs = np.array(N_xs)
    dxs=1/(N_xs+1)
    errors=[]
    
    if "HJ" in solver.__name__:
        analytic_solution = test_object.u
        final_timestep=-1
        error_type="infty"
    elif "FP" in solver.__name__:
        analytic_solution = test_object.m
        final_timestep=0
        error_type="K"
    
    for N_x in N_xs:
        test_object.set_N_x(N_x)
        
        numerical_solution = solver(g, **test_object.setup_dict)
        
        errors.append(_calc_rel_error(
            numerical_solution[:,final_timestep],
            analytic_solution(test_object.setup_dict["x"],
                              test_object.setup_dict["x"][final_timestep]),
            error_type
        ))

        if verbose>0: 
            print("The relative error is: {:.2e}".format(errors[-1]))
            print(f"Finished N_x={N_x}\n")

    error_name = "l^\infty" if error_type=="infty" else "\mathcal{K}"

    plt.figure(figsize=(6, 4.5))

    #Interpolation for numerical convergence rate
    res = linregress(log10(dxs[first_reg_node:last_reg_node]), log10(errors[first_reg_node:last_reg_node]))
    print(f"Numerical convergence rate: {signif(res.slope,3)}")
    slope = 0.5*round(res.slope/0.5) #Plot exponential slope rounded to nearest half.
    plt.loglog(dxs, 10**(res.intercept + slope*log10(dxs)),label="$h^{"+str(slope)+"}$",linestyle="--")

    #Plotting observed errors
    plt.loglog(dxs, errors, linestyle='', marker='o', label="Numerical results")
    plt.gca().invert_xaxis() #Inverting the x-axis and thus having largest stepsizes first
    if title: plt.title(title)
    plt.xlabel("Steplength $h$")
    ylabel = "Relative $"+f"{error_name}$ error"
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if filename:
        plt.savefig("Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()


# Functions for setting up the problem -----------------------------
def make_grid(N_x:int, t_end:np.float64, lamda=None, r=None)->dict:
    """Makes a grid in space and time. Defaulted to dt=h. dt is rounded as to get final timestep at t=t_end

    Args:
        N_x (int): Number of spatial discretization points.
        t_end (np.float64): Final time. Denoted in master's thesis as "T".
        lamda (np.float64, optional): Sets the temporal stepsize dt = lamda*h. Defaults to None.
        r (np.float64, optional): Sets the temporal stepsize dt = r*h^2. Defaults to None.

    Raises:
        Exception: If the user specifies both "r" and "lamda".

    Returns:
        dict: Returns a dictionary containing h, dt, N_t, x, t, grid_x, grid_t.
    """
    
    h = 1/(N_x)
    if lamda and r:
        raise Exception("Cannot specify both lamda and r in the initialization of the grid.")
    elif lamda:
        dt_temp = lamda*h
    elif r:
        dt_temp = r*h**2
    else:
        dt_temp = h
    
    N_t = ceil(t_end/dt_temp) #To get last value of discretized t on t_end
    dt = t_end/N_t

    #x = np.linspace(-dx,1+dx,N_x+3) #fictitous nodes kreves egentlig aldri! Brukes kun som en differanse med nest-siste/-første node, og da sier Homogen Neumann=0
    x = np.linspace(0,1,N_x+1)
    t = np.linspace(0,t_end,N_t+1)
    grid_x, grid_t = np.meshgrid(x,t,indexing="ij")
    
    return {"h":h, "dt":dt, "N_t":N_t, "x":x, "t":t, "grid_x":grid_x, "grid_t":grid_t}

def setup(N_x:int, t_end:np.float64, b:np.float64, mu_1:np.float64, u_0:Callable, 
          m_T:Callable, mu_0_func:Callable, f_HJ_func:Callable, 
          f_FP_func:Callable, lamda=None, r=None)->dict:
    """Constructs the discretized domain and calculates all function arguments on the discretized domain. Initializes the solutions M and U. Used in the constructor of the HJB and the MFG test problems in "test_problems.py".

    Args:
        N_x (int): Spatial steps.
        t_end (np.float64): Final time. Denotes "T" in the master's thesis.
        b (np.float64): Diffusion constant.
        mu_1 (np.float64): See form of HJB-equation.
        u_0 (Callable): Initial condition on u.
        m_T (Callable): Terminal condition on m.
        mu_0_func (Callable): See form of HJB-equation.
        f_HJ_func (Callable): See form of HJB-equation.
        f_FP_func (Callable): See form of FP-equation.
        lamda (np.float64, optional): Sets the temporal stepsize dt = lamda*h. Defaults to None.
        r (np.float64, optional): Sets the temporal stepsize dt = r*h^2. Defaults to None.

    Returns:
        dict: Dictionary containing h, dt, N_x, N_t, x, t, grid_x, grid_t, t_end, mu_0, f_HJ, f_FP, U_initial, U, M_terminal, M, b, mu_1.
    """
    h, dt, N_t, x, t, grid_x, grid_t = make_grid(N_x,t_end,lamda,r).values()
    
    shape = grid_x.shape
    mu_0 = mu_0_func(grid_x,grid_t)
    f_HJ = f_HJ_func(grid_x, grid_t)
    f_FP = f_FP_func(grid_x, grid_t)
    
    U_initial = u_0(x)
    U = np.zeros(shape)
    
    M_terminal = m_T(x)
    M,_ = np.meshgrid(M_terminal,t, indexing="ij")  #Initialize M as constant in time
    
    return {"h":h, "dt":dt, "N_x":N_x, "N_t":N_t, "x":x, "t":t, 
            "grid_x":grid_x, "grid_t":grid_t, "t_end":t_end,
            "mu_0":mu_0, "f_HJ":f_HJ, "f_FP":f_FP,
            "U_initial":U_initial, "U":U, "M_terminal":M_terminal,
            "M":M, "b":b, "mu_1":mu_1}
    
def setup_FP_test(N_x:int, lamda:np.float64, t_end:np.float64, b:np.float64, u:Callable, m_T:Callable, f_FP_func:Callable, **kwargs)->dict:
    """_summary_

    Args:
        N_x (int): Spatial steps
        lamda (np.float64): Sets the temporal stepsize dt = lamda*h
        t_end (np.float64): Final time. Denotes "T" in the master's thesis
        b (np.float64): Diffusion constant
        u (Callable): Value function u for use in a FP test-problem.
        m_T (Callable): Terminal condition on m
        f_FP_func (Callable): See form of FP-equation

    Returns:
        dict: Dictionary containing h, dt, N_x, N_t, x, t, grid_x, grid_t, t_end, f_HJ, U, M_terminal, M, b.
    """
    
    h, dt, N_t, x, t, grid_x, grid_t = make_grid(N_x,t_end,lamda).values()
    
    f_FP = f_FP_func(grid_x, grid_t)
    
    shape = grid_x.shape
    U = u(grid_x, grid_t)
    
    M_terminal = m_T(x)
    M = np.zeros(shape)
    
    return {"h":h, "dt":dt, "N_x":N_x, "N_t":N_t, "x":x, "t":t, "grid_x":grid_x, 
            "grid_t":grid_t, "t_end":t_end, "f_FP":f_FP, "U":U,  
            "M_terminal":M_terminal,"M":M, "b":b}

# Numerical Hamiltonians and their derivatives ----------------------
def godunov(q_1:np.array,q_2:np.array)->np.array:
    """The godunov-type numerical Hamiltonian for H(x,p)=p^2."""
    return np.maximum(-q_1,0)**2 + np.maximum(q_2,0)**2

def _godunov_q_1(q_1:np.array,q_2:np.array)->np.array:
    """The partial derivative for the godunov-type numerical Hamiltonian w.r.t. q_1 for H(x,p)=p^2"""
    return -2*np.maximum(-q_1,0)

def _godunov_q_2(q_1:np.array,q_2:np.array)->np.array:
    """The partial derivative for the godunov-type numerical Hamiltonian w.r.t. q_2 for H(x,p)=p^2"""
    return 2*np.maximum(q_2,0)

# Numerical solvers -------------------------------------------------
def _HJ_scheme_implicit(U_internal:np.array, g:Callable, U_internal_prev:np.array, mu:np.array, f:np.array, dt:np.float64, h:np.float64, b:np.float64)->np.array:
    """Calculates the left hand side of the HJB-equation for the internal points at some timestep t^n. To be solved by "scipy.optimize.fsolve" for finding "U^{n+1}"=U_internal

    Args:
        U_internal (np.array): [U^{n+1}_{1},...,U^{n+1}_{N_x-1}]
        g (Callable): Numerical Hamiltonian.
        U_internal_prev (np.array): [U^{n}_{1},...,U^{n}_{N_x-1}]
        mu (np.array): Discretized mu for the internal points at timestep t^{n+1}.
        f (np.array): Discretized f for the internal points at timestep t^{n+1}.
        dt (np.float64): Temporal stepsize.
        h (np.float64): Spatial stepsize.
        b (np.float64): Diffusion coefficient.

    Returns:
        np.array: Left hand side of the HJB-equation for the internal points at some timestep t^n.
    """
    return ((U_internal-U_internal_prev)/dt 
            +g( (roll(U_internal,-1)-U_internal)/h, 
                (U_internal-roll(U_internal,1))/h ) 
            - b*(roll(U_internal,-1)- 2*U_internal + roll(U_internal,1) )/h**2 
            + mu*U_internal +f )

def _HJ_scheme_implicit_step(g:Callable, U_internal_start_guess:np.array, U__internal_prev:np.array, 
                             mu:np.array, f:np.array, dt:np.float64, h:np.float64, b:np.float64)->np.array:
    """Does one step of the HJB-scheme. That is, it solves the system of non-linear equations that the HJB-equation reduces to at timestep t^n.

    Args:
        g (Callable): Numerical Hamiltonian.
        U_internal_start_guess (np.array): Initial guess for the solution, as we solve the system of nonlinear equations by the means of a iterative method.
        U__internal_prev (np.array): [U^{n}_{1},...,U^{n}_{N_x-1}]
        mu (np.array): Discretized mu for the internal points at timestep t^{n+1}
        f (np.array): Discretized f for the internal points at timestep t^{n+1}
        dt (np.float64): Temporal stepsize.
        h (np.float64): Spatial stepsize.
        b (np.float64): Diffusion coefficient.

    Returns:
        np.array: [U^{n+1}_{1},...,U^{n+1}_{N_x-1}]
    """
    return fsolve(_HJ_scheme_implicit, U_internal_start_guess,
                args=(g, U__internal_prev, mu, f, dt, h, b),
                xtol=10**-10)
    
    # Experimental
    return fsolve(_HJ_scheme_implicit, U_internal_start_guess,
            args=(g, U__internal_prev, mu, f, dt, h, b),
            xtol=10**-10, fprime=_HJ_implicit_Jacobian)

def HJ_implicit(g:Callable, h:np.float64, dt:np.float64, N_t:int, mu_0:np.array, f_HJ:np.array, U_initial:np.array, U:np.array, M:np.array, b:np.float64, mu_1:np.float64,**kwargs)->np.array:
    """Solves the discretized HJB-equation at all the timesteps using the implicit method described in the master's thesis.

    Args:
        g (Callable): Numerical Hamiltonian.
        h (np.float64): Spatial stepsize.
        dt (np.float64): Temporal stepsize.
        N_t (int): Number of steps in time.
        mu_0 (np.array): mu_0 on the whole discretized domain. mu_0 as given in the HJB-equation.
        f_HJ (np.array): f_HJ on the whole discretized domain. f_HJ as given in the HJB-equation.
        U_initial (np.array): Initial value of u on the spatial discretization.
        U (np.array): Initialization of U on the discretized domain.
        M (np.array): M on the discretized domain.
        b (np.float64): Diffusion coefficient.
        mu_1 (np.float64): As given in the HJB-equation

    Returns:
        np.array: The solution U of the discretized HJB-equation.
    """
    U[:,0] = U_initial
    for i in range(N_t):
        # Solves the fixed-point problem to find the internal points with homogenous Neumann BCs
        U[1:-1,i+1] = _HJ_scheme_implicit_step(g, U[1:-1,i], U[1:-1,i], 
                                            (mu_0[1:-1,i+1]+mu_1*M[1:-1,i+1]),
                                            f_HJ[1:-1,i+1], dt, h, b)
        U[0,i+1] = U[1,i+1] # Homogenous Neumann BC
        U[-1,i+1] = U[-2,i+1] # Homogenous Neumann BC
    return U

def _FP_create_A(U_vec:np.array, g_q_1:Callable, g_q_2:Callable, b:np.float64, 
                 dt:np.float64, h:np.float64) -> scipy.sparse._dia.dia_matrix:
    """Creates the sparse matrix A such that M^{n+1}= A M^{n}.

    Args:
        U_vec (np.array): U at timestep t^n
        g_q_1 (Callable): Partial derivative of g w.r.t. q_1.
        g_q_2 (Callable): Partial derivative of g w.r.t. q_2.
        b (np.float64): Diffusion coefficient.
        dt (np.float64): Temporal stepsize.
        h (np.float64): Spatial stepsize.

    Returns:
        scipy.sparse._dia.dia_matrix: Matrix A
    """
    
    g_q_1_vec = g_q_1( (roll(U_vec,-1)-U_vec)/h, 
                             (U_vec - roll(U_vec, 1))/h )
    g_q_2_vec = g_q_2( (roll(U_vec,-1)-U_vec)/h, 
                             (U_vec - roll(U_vec, 1))/h )
    upper = -dt/h * (g_q_2_vec + b/h)
    upper[1] *= 2 #As first interval is half-length
    
    lower = -dt/h * (-g_q_1_vec + b/h)
    lower[-2]*= 2 #As last interval is half-length
    
    middle = 1 + dt/h * (-g_q_1_vec + g_q_2_vec + 2*b/h)
    middle[0]  = 1 + dt / (h/2) * (-g_q_1_vec[0] + b/h)
    middle[-1] = 1 + dt / (h/2) * (g_q_2_vec[-1]+ b/h)
    
    return scipy.sparse.spdiags([upper, middle, lower],[1,0,-1])

def _FP_implicit_step(U_vec:np.array, M_next:np.array, f:np.array, g_q_1:Callable, g_q_2:Callable, b:np.float64, dt:np.float64, h:np.float64, 
                      calculate_inverse=False, **kwargs)->np.array:
    """Does one step of the FP-scheme. That is, That is, it solves the discretized FP-equation at all the timesteps using the implicit method described in the master's thesis.

    Args:
        U_vec (np.array): U at timestep t^n
        M_next (np.array): M at timestep t^{n+1}
        f (np.array): f on the spatial discretization at timestep t^{n+1}
        g_q_1 (Callable): Partial derivative of g w.r.t. q_1.
        g_q_2 (Callable): Partial derivative of g w.r.t. q_2.
        b (np.float64): Diffusion coefficient.
        dt (np.float64): Temporal stepsize.
        h (np.float64): Spatial stepsize.
        calculate_inverse (bool, optional): Choose whether to solve the scheme using a linear function solver or by calculating the inverse of A. Defaults to False.

    Returns:
        _type_: M at timestep t^n
    """
    if calculate_inverse:
        A = _FP_create_A(U_vec, g_q_1, g_q_2, b, dt, h).toarray()
        return scipy.linalg.inv(A) @ (M_next + dt*f)
    
    else:
        A = _FP_create_A(U_vec, g_q_1, g_q_2, b, dt, h)
        return scipy.sparse.linalg.bicgstab(A , M_next + dt*f)[0]

def FP_implicit(g:Callable, M_terminal:np.array, M:np.array, U:np.array, 
                f_FP:np.array, b:np.float64, h:np.float64, dt:np.float64, N_t:int, 
                **kwargs) -> np.array:
    """Solves the discretized FP-equation at all the timesteps using the implicit method described in the master's thesis.

    Args:
        g (Callable): Numerical Hamiltonian.
        M_terminal (np.array): Terminal value of m on the spatial discretization.
        M (np.array): Initialization of M on the discretized domain.
        U (np.array): U on the discretized domain.
        f_FP (np.array): f_FP on the whole discretized domain. f_HJ as given in the HJB-equation.
        b (np.float64): Diffusion coefficient.
        h (np.float64): Spatial stepsize.
        dt (np.float64): Temporal stepsize.
        N_t (int): Number of steps in time.

    Returns:
        np.array: The solution M of the discretized FP-equation.
    """

    if g.__name__=="godunov":
        g_q_1 = _godunov_q_1
        g_q_2 = _godunov_q_2
    
    M[:,N_t] = M_terminal
    for n in range(N_t,0,-1):
        M[:,n-1] = _FP_implicit_step(U[:,n-1], M[:,n], f_FP[:,n-1], 
                                     g_q_1, g_q_2, b, dt, h, **kwargs)
    
    return M

def MFG_solver(HJ_solver:Callable, FP_solver:Callable, g:Callable, U:np.array, U_initial:np.array, M:np.array, M_terminal:np.array,
               mu_0:np.array, f_HJ:np.array, f_FP:np.array, b:np.array, mu_1:np.float64, h:np.float64, dt:np.float64, N_t:int, t_end:int, grid_x:np.array, grid_t:np.array, 
               m=None, u=None, a=1/2, plot_freq=200, iterations=100, 
               stop=10**-3, filename=None, **kwargs)->tuple[np.array, np.array]:
    """Solves the discretized MFG system using the fixed point iteration described in my master's thesis using "HJ_solver" to solve the discretized HJB-equation at each iteration and "FP_solver" to solve the discretized FP-equation at each iteration.

    Args:
        HJ_solver (Callable): HJB-scheme
        FP_solver (Callable): FP-scheme
        g (Callable): Numerical Hamiltonian
        U (np.array): Initialization of the value function u on the discretized domain.
        U_initial (np.array): Initial value of u on the spatial discretization.
        M (np.array): Initialization of the probability density m on the discretized domain.
        M_terminal (np.array): Terminal value of m on the spatial discretization.
        mu_0 (np.array): mu_0 on the whole discretized domain. mu_0 as given in the HJB-equation.
        f_HJ (np.array): f_HJ on the whole discretized domain. f_HJ as given in the HJB-equation.
        f_FP (np.array): f_FP on the whole discretized domain. f_FP as given in the FP-equation.
        b (np.array): Diffusion coefficient.
        mu_1 (np.float64): As given in the HJB-equation
        h (np.float64): Spatial stepsize.
        dt (np.float64): Temporal stepsize.
        N_t (int): Number of steps in time.
        t_end (int): Final stepsize. Denoted "T" in the master's thesis.
        grid_x (np.array): The discrete x-values on the whole grid. (N_x+1)x(N_t+1)-matrix. Produced by "np.meshgrid" in "setup".
        grid_t (np.array): The discrete t-values on the whole grid. (N_x+1)x(N_t+1)-matrix. Produced by "np.meshgrid" in "setup".
        m (Callable, optional): The analytic solution m of the FP-equation. Defaults to None.
        u (Callable, optional): The analytic solution u of the HJB-equation.. Defaults to None.
        a (np.float64, optional): Memory, as given in the fixed point iteration. Defaults to 1/2.
        plot_freq (int, optional): How often to plot the status of the fixed point iteration. Defaults to 200.
        iterations (int, optional): Maximum number of iterations before terminating the fixed point iteration. Defaults to 100.
        stop (np.float, optional): Tolerance for the fixed point iteration. It stops when the relative change in U and M is less than "stop". Defaults to 10**-3.
        filename (str, optional): Saves the final iteration-status-plot under "/Images/<filename>.pdf". Defaults to None..

    Returns:
        tuple[np.array, np.array]: Return the solution (U,M) of the discretized MFG system.
    """
    
    #Track change in U, M for stopping criterion
    U_change=[]
    M_change=[]
    
    #Track approximation error if have analytic function
    have_analytic = m!=None and u!=None
    if have_analytic:
        U_error =[]
        M_error =[]
        u_vec = u(grid_x,grid_t)
        m_vec = m(grid_x,grid_t)
    else:
        U_error=None
        M_error=None
        
    for i in range(iterations):
        #Next iteration of fixed point method
        U_new = HJ_solver(g, h, dt, N_t, mu_0, f_HJ, U_initial,np.copy(U),M,b,mu_1)
        M_new = FP_solver(g, M_terminal, np.copy(M), U_new, f_FP, b, h, dt, N_t,**kwargs)
        
        #Calculate and track change in U, M
        U_change.append(_calc_rel_error(U[:,:],
                                        U_new[:,:],
                                        "infty"))
        M_change.append(_calc_rel_error(M[:,:],
                                        M_new[:,:],
                                        "K"))
        
        #Tracking approximation error
        if have_analytic:
            U_error.append(_calc_rel_error(U_new[:,-1],u_vec[:,-1],
                                        errortype="infty"))
            M_error.append(_calc_rel_error(M_new[:,0],m_vec[:,0],
                                        errortype="K"))
        
        #Status-plotting
        if i%plot_freq==plot_freq-1:
            _MFG_solver_plot_convergence_of_fixed_point_iterations(U_change, 
                                                                    M_change,
                                                                    a,
                                                                    U_error,
                                                                    M_error)
        
        #Update U, M
        if i==0: #Completely update the arbitrarily chosen initialization of U, M
            U, M = U_new, M_new
        else: #Some convex combination of new and old value
            U, M = U_new , a*M + (1-a)*M_new
            #U, M = lamda*U + (1-lamda)*U_new , lamda*M + (1-lamda)*M_new

        #Stop loop if change in M and U are small
        if (U_change[-1] < stop) and (
            M_change[-1] < stop):
            print(f"Stopped due to very small change in resulting matrices.\niteration #{i+1}")
            break
    
    #Status-plotting
    _MFG_solver_plot_convergence_of_fixed_point_iterations(U_change, M_change, a,
                                                           U_error, M_error,
                                                           filename=filename)
    
    
    return U,M


# Experimental --------------------------------------------------------
def _middle_of_interval_indices(N_t:int,t_end:np.float64)->tuple[int,int]:
    start_index = int(N_t/t_end * (t_end)/2)
    stop_index = int(N_t/t_end * (t_end+2)/2)
    return start_index,stop_index

def plot_middle_day(numerical:np.array, N_t:int, t_end:np.float64, grid_x:np.array, grid_t:np.array, title,
                    analytic=None, filename=None, **kwargs)->None:
    """Plots heatmap of the numerical mass density M/value function U over time in the middle of the time-domain [n,n+1] for a positive integer n. May also plot the analytic solution m.

    Args:
        numerical (np.array): M or U over time. (N_x+1)x(N_t+1)-matrix
        N_t (int): Final timestep.
        t_end (np.float64): Final time. Denoted T in the master's thesis.
        grid_x (np.array): The discrete x-values on the whole grid. (N_x+1)x(N_t+1)-matrix. Produced by "np.meshgrid" in "setup"
        grid_t (np.array): The discrete t-values on the whole grid. (N_x+1)x(N_t+1)-matrix. Produced by "np.meshgrid" in "setup"
        title str: Name to be used in plot titles of format "functionname $f$". E.g. "mass density $m$". 
        analytic (Callable, optional): The analytic function (m or u) as a funtion. Defaults to None.
        filename (str, optional): Saves the plots under "/Images/<filename>.pdf". Defaults to None.
    """
    start_index, stop_index = _middle_of_interval_indices(N_t,t_end)
    x=grid_x[:,start_index:stop_index]
    t=grid_t[:,start_index:stop_index]
    numerical_crop = numerical[:,start_index:stop_index]
    min_val = np.min(numerical_crop)
    max_val = np.max(numerical_crop)
    extent=[t.min(), t.max(), x.max(),x.min()]
    
    if analytic!=None:
        analytic_vals = analytic(x,t)
        min_val = min(min_val, np.min(analytic_vals))
        max_val = max(max_val, np.max(analytic_vals))
        
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax2.set_title("Analytic "+title+ " one day")
        ax2.imshow(analytic_vals, cmap="coolwarm", vmin=min_val, vmax=max_val)
    else:
        fig,ax1 = plt.subplots()
        
    ax1.set_title("Numerical "+title+" one day")
    im = ax1.imshow(numerical_crop, cmap="coolwarm", vmin=min_val, vmax=max_val,
                    extent=extent, aspect="auto")
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if filename:
        plt.savefig("Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()

def _HJ_implicit_Jacobian(U, g, U_prev, mu, f, dt, h, b) -> scipy.sparse._dia.dia_matrix:
    
    g_q_1_vec = _godunov_q_1( (roll(U,-1)-U)/h, 
                             (U - roll(U, 1))/h )
    g_q_2_vec = _godunov_q_2( (roll(U,-1)-U)/h, 
                             (U - roll(U, 1))/h )
    
    upper = -1/h * (g_q_2_vec + b/h)
    lower = -1/h * (-g_q_1_vec + b/h)
    
    middle = 1/dt + 1/h * (-g_q_1_vec + g_q_2_vec + 2*b/h) + mu
    middle[0]  = 1/dt + 1/h * (-g_q_1_vec[0] + b/h) + mu[0]
    middle[-1] = 1/dt + 1/h * (g_q_2_vec[-1]+ b/h) + mu[-1]
    
    return scipy.sparse.spdiags([upper, middle, lower],[1,0,-1]).toarray()

def _HJ_explicit_create_A(U_length, b,h,dt):
    upper = b*dt/h**2 + np.zeros(U_length)
    lower = b*dt/h**2 + np.zeros(U_length)
    
    middle = 1 - 2*b*dt/h**2 + np.zeros(U_length)
    middle[0]  = 1 - b*dt/h**2
    middle[-1] = 1 - b*dt/h**2
    
    return scipy.sparse.spdiags([upper, middle, lower],[1,0,-1])

def _HJ_scheme_explicit_step(A, g, U_prev, mu, f, dt, h, b):
    cfl_cond = 1/(2*b + 2*h*np.max(abs(roll(U_prev,-1)-U_prev)/h) + h**2 * np.max(mu) )
    if dt/h**2 >= cfl_cond:
        raise Exception(f"CFL condition is not satisfied. r={dt/h**2}!<{cfl_cond}")
    
    return (A@U_prev - dt*g( (roll(U_prev,-1)-U_prev)/h, (U_prev-roll(U_prev,1))/h ) 
            -dt*f - dt*mu*U_prev)
    
def HJ_explicit(g, h, dt, N_t, mu_0, f_HJ, U_initial, U, M, b, mu_1,**kwargs):
    A = _HJ_explicit_create_A(U_initial.shape[0],b,h,dt)
    U[:,0] = U_initial
    for i in range(N_t):
        U[:,i+1] = _HJ_scheme_explicit_step(A, g, U[:,i], 
                                            (mu_0[:,i]+mu_1*M[:,i]),
                                            f_HJ[:,i], dt, h, b) 
    return U
