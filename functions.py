import scipy
from scipy.stats import linregress
from scipy.integrate import simpson
from scipy.optimize import fsolve, broyden1, broyden2
import numpy as np
from math import ceil
from numpy import log10
import matplotlib.pyplot as plt
from inspect import isfunction
from sys import getsizeof
from typing import Callable


def signif(x, p): 
    """Writes x with p number of significant digits"""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def namestr(obj, namespace):
    """Stack Overflow magic. Returns name of array/variable."""
    return [name for name in namespace if namespace[name] is obj][0]

def calc_errors(numeric_sol, analytic_fnc, setup_dict, errortype):
    
    analytic_sol = analytic_fnc(setup_dict["grid_x"], setup_dict["grid_t"])
    numeric_sol_terminal = numeric_sol[:,-1]
    analytic_sol_terminal = analytic_sol[:,-1]
    
    if errortype=="infty":
        terminal_error_abs = np.max(np.abs(numeric_sol_terminal-analytic_sol_terminal))
        terminal_error_rel = terminal_error_abs/np.max(np.abs(analytic_sol_terminal))
        
        everywhere_error_abs = np.max(np.abs(numeric_sol-analytic_sol))
        everywhere_error_rel = everywhere_error_abs/np.max(np.abs(analytic_sol))
        
        max_error_abs = np.max(np.abs(numeric_sol-analytic_sol))
        max_error_rel = everywhere_error_abs/np.max(np.abs(analytic_sol))
        
    elif errortype=="L1":
        terminal_error_abs = simpson(np.abs(numeric_sol_terminal-analytic_sol_terminal),
                                     dx=setup_dict["h"])
        terminal_error_rel = terminal_error_abs/np.simpson(np.abs(analytic_sol_terminal),
                                                           dx=setup_dict["h"])
        
        everywhere_error_abs = simpson(simpson(np.abs(numeric_sol-analytic_sol),
                                       dx=setup_dict["h"]),dx=setup_dict["dt"])
        everywhere_error_rel = everywhere_error_abs/simpson(simpson(np.abs(analytic_sol),
                                       dx=setup_dict["h"]),dx=setup_dict["dt"])
        
        max_error_abs = np.max(simpson(np.abs(numeric_sol-analytic_sol),
                                       dx=setup_dict["h"]))
        max_error_rel = everywhere_error_abs/np.max(simpson(np.abs(analytic_sol),
                                       dx=setup_dict["h"]))
        
    
    return {"terminal_error_abs":terminal_error_abs, 
            "terminal_error_rel":terminal_error_rel,
            "everywhere_error_abs":everywhere_error_abs,
            "everywhere_error_rel":everywhere_error_rel,
            "max_error_abs":max_error_abs,
            "max_error_rel":max_error_rel}

def K_norm(M):
    if len(M.shape)==1:
        return (np.abs(M[0])/2 + np.abs(M[-1])/2 + np.sum(np.abs(M[1:-1])) )/(M.shape[0]-1)
    elif len(M.shape)==2:
        return np.array([ (M[0,i]/2 + M[-1,i]/2 + np.sum(M[1:-1,i])) 
                           for i in range(M.shape[1])])/(M.shape[0]-1)
    else:
        raise Exception(f"K_norm only supports vectors or matrices. Input has shape {M.shape}")

def _calc_rel_error(numerical, analytic, errortype):
    """Calculates the relative difference between numerical and analytic with the norm given by errortype."""
    if errortype=="infty":
        return np.max(np.abs(numerical-analytic))/np.max(np.abs(analytic))
    elif errortype=="K":
        return np.max(K_norm(numerical-analytic))/np.max(K_norm(analytic))
        #return simpson(np.abs(numerical-analytic))/simpson(np.abs(analytic))
        #return np.sum(np.abs(numerical-analytic))/np.sum(np.abs(analytic))
    else:
        raise Exception("Only errortypes 'infty' and 'K' are supported.")

def _middle_of_interval_indices(N_t,t_end):
    start_index = int(N_t/t_end * (t_end)/2)
    stop_index = int(N_t/t_end * (t_end+2)/2)
    return start_index,stop_index
    
def plot_different_times(numeric_sol, analytic_fnc, setup_dict, 
                         title=None,filename=None, ylabel="y",**subplots_kwargs):
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
        plt.savefig("../Thesis/Figures/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()

def plot_total_mass(h, t, M, filename=None, **kwargs):
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
        plt.savefig("../Thesis/Figures/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()

def plot_mass_one_day(M_numerical, N_t, t_end, grid_x, grid_t, m=None, **kwargs):
    start_index, stop_index = _middle_of_interval_indices(N_t,t_end)
    x=grid_x[:,start_index:stop_index]
    t=grid_t[:,start_index:stop_index]
    M_crop = M_numerical[:,start_index:stop_index]
    m_min = np.min(M_crop)
    m_max = np.max(M_crop)
    
    if m!=None:
        m_vals = m(x,t)
        m_min = min(m_min, np.min(m_vals))
        m_max = max(m_max, np.max(m_vals))
        
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax2.set_title("Analytic mass $m$ one day")
        ax2.imshow(m_vals, cmap="coolwarm", vmin=m_min, vmax=m_max)
    else:
        fig,ax1 = plt.subplots()
        
    ax1.set_title("Numerical mass $M$ one day")
    im = ax1.imshow(M_crop, cmap="coolwarm", vmin=m_min, vmax=m_max)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    
def plot_middle_day(numerical, N_t, t_end, grid_x, grid_t, analytic=None, 
                    title="function $f$", filename=None, **kwargs):
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
        plt.savefig("../Thesis/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()

def _MFG_solver_plot_convergence_of_fixed_point_iterations(U_error, M_error,
                                                           U_change, M_change,
                                                           lamda,  
                                                           have_analytic, 
                                                           filename =None):
    markersize=3.5
    if have_analytic:
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
    fig.suptitle("Evolution of fixed point iteration, $\lambda=$"+f"{signif(lamda,2)}")
    ax1.plot(U_change,label="Relative $L^\infty$ change in U",
                marker="o", markersize=markersize)
    ax1.plot(M_change, label="Relative $\mathcal{K}$ change in M",
                marker="o", markersize=markersize)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid()
    
    if filename:
        plt.savefig("../Thesis/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()

def loglog(solver, N_xs, test_object, g, first_reg_node=0, 
           last_reg_node=-1, verbose=1, title=None, filename=None):
    """Makes a loglog plot.
    
    Parameters
    ----------
    solver: function
        Function for solving the problem.
    arguments: Dict
        Dictionary containing input data for the solver.
    N_xs: list
        List with number of spatial steps to be considered.
    first_reg_node: int, optional
        Regression will be done on the error points [first_reg_node,last_reg_node)
    last_reg_node: int, optional
        Regression will be done on the error points [first_reg_node,last_reg_node)
    errortype: string, optional
        "infty" or "K", determining which error term to use.
    verbose: int, optional
        0: no printing, >0: print information continuously
    title: string, optional
        If given, the plot will be given this title.
    filename: string, optional
        If given, the plot will be saved as a pdf with this name.
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
        plt.savefig("../Thesis/Figures/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")  
    plt.show()

def loglog_MFG( N_xs, test_object, g, fixed_point_stop=10**-3, lamda=1/3, 
               first_reg_node=0, last_reg_node=-1, verbose=1, U_title=None,
               M_title=None, filename=None, filename_iterations=None):
    N_xs = np.array(N_xs)
    dxs=1/(N_xs+1)
    U_errors=[]
    M_errors=[]
    
    u=test_object.u 
    m=test_object.m 
    
    for N_x in N_xs:
        if filename_iterations:
            filename_iter = filename_iterations+"_"+str(N_x)
        else:
            filename_iter = None
        
        test_object.set_N_x(N_x)
        U, M = MFG_solver(HJ_implicit, FP_implicit, g, 
                                        **test_object.setup_dict, u=u, m=m,
                                        stop=fixed_point_stop, lamda=lamda,
                                        filename=filename_iter)
        
        U_errors.append(_calc_rel_error(
            U[:,-1],
            u(test_object.setup_dict["x"], test_object.setup_dict["x"][-1]),
            "infty"
        ))
        M_errors.append(_calc_rel_error(
            M[:,0],
            m(test_object.setup_dict["x"], test_object.setup_dict["x"][0]),
            "K"
        ))

        if verbose>0: 
            print(f"N_x={N_x}".ljust(15),end="")
            print("Error M: {:.2e}".format(M_errors[-1]).ljust(25),end="")
            print("Error U: {:.2e}".format(U_errors[-1]))
            


    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6, 4.5))
    
    #U
    #Interpolation for numerical convergence rate
    res_U = linregress(log10(dxs[first_reg_node:last_reg_node]), 
                     log10(U_errors[first_reg_node:last_reg_node]))
    print(f"Numerical convergence rate: {signif(res_U.slope,3)}")
    
    #Plot exponential slope rounded to nearest half.
    slope_U = 0.5*round(res_U.slope/0.5) 
    ax1.loglog(dxs, 10**(res_U.intercept + slope_U*log10(dxs)),label="$h^{"+str(slope_U)+"}$",linestyle="--")
    
    #M
    #Interpolation for numerical convergence rate
    res_M = linregress(log10(dxs[first_reg_node:last_reg_node]), 
                     log10(M_errors[first_reg_node:last_reg_node]))
    print(f"Numerical convergence rate: {signif(res_M.slope,3)}")
    
    #Plot exponential slope rounded to nearest half.
    slope_M = 0.5*round(res_M.slope/0.5) 
    ax2.loglog(dxs, 10**(res_M.intercept + slope_M*log10(dxs)),label="$h^{"+str(slope_M)+"}$",linestyle="--")

    #Plotting observed errors
    ax1.loglog(dxs, U_errors, linestyle='', marker='o', label="Numerical results")
    ax1.invert_xaxis() #Inverting the x-axis and thus having largest stepsizes first
    if U_title: ax1.suptitle(U_title)
    ax1.set_xlabel("Steplength $h$")
    ax1.set_ylabel("Relative $l^\infty$ error")
    ax1.legend()
    ax1.grid()
    
    ax2.loglog(dxs, M_errors, linestyle='', marker='o', label="Numerical results")
    ax2.invert_xaxis() #Inverting the x-axis and thus having largest stepsizes first
    if M_title: ax2.suptitle(M_title)
    ax2.set_xlabel("Steplength $h$")
    ax2.set_ylabel("Relative $L^1$ error")
    ax2.legend()
    ax2.grid()
    
    
    if filename:
        plt.savefig("../Thesis/Images/"+filename+".pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()
    
def make_grid(N_x, t_end, lamda=None, r=None):
    """Makes a grid in space and time. x has one ficitious node at each end. dt is rounded as to get final timestep at t=t_end"""
    
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

def godunov(q_1,q_2):
    return np.maximum(-q_1,0)**2 + np.maximum(q_2,0)**2

def lax_friedrich(q_1,q_2):
    return ((q_1+q_2)/2)**2-10*(q_1-q_2)

def roll(a, shift):
    """np.roll but instead of having a "circular" array, elements that roll beyond the last position disappear and the missing value stays as it did before."""
    ret = np.roll(a,shift)
    if shift>0:
        ret[0]=a[0]
    elif shift<0:
        ret[-1]=a[-1]
    return ret

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

def _HJ_scheme_implicit(U_internal, g, U_internal_prev, mu, f, dt, h, b):
    return ((U_internal-U_internal_prev)/dt 
            +g( (roll(U_internal,-1)-U_internal)/h, 
                (U_internal-roll(U_internal,1))/h ) 
            - b*(roll(U_internal,-1)- 2*U_internal + roll(U_internal,1) )/h**2 
            + mu*U_internal +f )

def _HJ_scheme_implicit_step(g, U_internal_start_guess, U__internal_prev, 
                             mu, f, dt, h, b):
    return fsolve(_HJ_scheme_implicit, U_internal_start_guess,
                args=(g, U__internal_prev, mu, f, dt, h, b),
                xtol=10**-10)
    return fsolve(_HJ_scheme_implicit, U_internal_start_guess,
            args=(g, U__internal_prev, mu, f, dt, h, b),
            xtol=10**-10, fprime=_HJ_implicit_Jacobian)
    return broyden2(lambda x: _HJ_scheme_implicit(x, g, U__internal_prev, mu, 
                                                  f, dt, h, b),
                    U_internal_start_guess, reduction_method="svd", verbose=1,
                    line_search="wolfe")
    return fsolve(lambda x: _HJ_scheme_implicit(x, g, U__internal_prev, mu, 
                                                f, dt, h, b), 
                  U_internal_start_guess, full_output=True)

def HJ_implicit(g, h, dt, N_t, mu_0, f_HJ, U_initial, U, M, b, mu_1,**kwargs):
    U[:,0] = U_initial
    for i in range(N_t):
        # Solves the fixed-point problem to find the internal points with homogenous Neumann BCs
        U[1:-1,i+1] = _HJ_scheme_implicit_step(g, U[1:-1,i], U[1:-1,i], 
                                            (mu_0[1:-1,i+1]+mu_1*M[1:-1,i+1]),
                                            f_HJ[1:-1,i+1], dt, h, b)
        U[0,i+1] = U[1,i+1] # Homogenous Neumann BC
        U[-1,i+1] = U[-2,i+1] # Homogenous Neumann BC
    return U

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

def setup(N_x, t_end, b, mu_1, u_0, m_T, mu_0_func, f_HJ_func, 
          f_FP_func, lamda=None, r=None):
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
    
def setup_FP_test(N_x, lamda, t_end, b, u, m_T, f_FP_func, **kwargs):
    h, dt, N_t, x, t, grid_x, grid_t = make_grid(N_x,t_end,lamda).values()
    
    f_FP = f_FP_func(grid_x, grid_t)
    
    shape = grid_x.shape
    U = u(grid_x, grid_t)
    
    M_terminal = m_T(x)
    M = np.zeros(shape)
    
    return {"h":h, "dt":dt, "N_t":N_t, "x":x, "t":t, "grid_x":grid_x, 
            "grid_t":grid_t, "f_FP":f_FP, "U":U, "M":M, 
            "M_terminal":M_terminal,"b":b}

def _godunov_q_1(q_1,q_2):
    return -2*np.maximum(-q_1,0)

def _godunov_q_2(q_1,q_2):
    return 2*np.maximum(q_2,0)

def _FP_create_A(U_vec:np.array, g_q_1:Callable, g_q_2:Callable, b:float, 
                 dt:float, h:float) -> scipy.sparse._dia.dia_matrix:
    
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

def _FP_implicit_step(U_vec, M_next, f, g_q_1, g_q_2, b, dt, h, 
                      calculate_inverse=False, **kwargs):
    
    if calculate_inverse:
        A = _FP_create_A(U_vec, g_q_1, g_q_2, b, dt, h).toarray()
        return scipy.linalg.inv(A) @ (M_next + dt*f)
    
    else:
        A = _FP_create_A(U_vec, g_q_1, g_q_2, b, dt, h)
        #return scipy.sparse.linalg.cg(A , M_next + dt*f, atol=10**-5)[0]
        return scipy.sparse.linalg.bicgstab(A , M_next + dt*f)[0]

def FP_implicit(g:Callable, M_terminal:np.array, M:np.array, U:np.array, 
                f_FP:np.array, b:float, h:float, dt:float, N_t:int, 
                **kwargs) -> np.array:
    if g.__name__=="godunov":
        g_q_1 = _godunov_q_1
        g_q_2 = _godunov_q_2
    
    M[:,N_t] = M_terminal
    for n in range(N_t,0,-1):
        M[:,n-1] = _FP_implicit_step(U[:,n-1], M[:,n], f_FP[:,n-1], 
                                     g_q_1, g_q_2, b, dt, h, **kwargs)
    
    return M

def MFG_solver(HJ_solver, FP_solver, g, U, U_initial, M, M_terminal,
               mu_0, f_HJ, f_FP, b, mu_1, h, dt, N_t, t_end, grid_x, grid_t, 
               m=None, u=None, lamda=1/3, plot_freq=200, iterations=100, 
               stop=10**-3, filename=None, type=1, check_mid=False, **kwargs):
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
        
        # Type 1: M -> U_new -> M_new, men så M,U->lambda M +(1-lambda)M_new, samme greia
        if type==1:
            #Next iteration of fixed point method
            U_new = HJ_solver(g, h, dt, N_t, mu_0, f_HJ, U_initial,np.copy(U),M,b,mu_1)
            M_new = FP_solver(g, M_terminal, np.copy(M), U_new, f_FP, b, h, dt, N_t,**kwargs)
            
            #Choose what times to calculate change and error
            if check_mid:
                start_index, stop_index = _middle_of_interval_indices(N_t,t_end)
            else:
                start_index, stop_index = 0,None
            
            #Calculate and track change in U, M
            U_change.append(_calc_rel_error(U[:,start_index:stop_index],
                                            U_new[:,start_index:stop_index],
                                            "infty"))
            M_change.append(_calc_rel_error(M[:,start_index:stop_index],
                                            M_new[:,start_index:stop_index],
                                            "K"))
            
            #Tracking approximation error
            if have_analytic:
                U_error.append(_calc_rel_error(U_new[:,-1],u_vec[:,-1],
                                            errortype="infty"))
                M_error.append(_calc_rel_error(M_new[:,0],m_vec[:,0],
                                            errortype="K"))
            
            #Status-plotting
            if i%plot_freq==plot_freq-1:
                _MFG_solver_plot_convergence_of_fixed_point_iterations(U_error,
                                                                       M_error,
                                                                       U_change, 
                                                                       M_change, 
                                                                       lamda, 
                                                                       have_analytic)
            
            #Update U, M
            if i==0: #Completely update the arbitrarily chosen initialization of U, M
                U, M = U_new, M_new
            else: #Some convex combination of new and old value
                U, M = U_new , lamda*M + (1-lamda)*M_new
                #U, M = lamda*U + (1-lamda)*U_new , lamda*M + (1-lamda)*M_new

            #Stop loop if change in M and U are small
            if (U_change[-1] < stop) and (
                M_change[-1] < stop):
                print(f"Stopped due to very small change in resulting matrices.\niteration #{i+1}")
                break
        
        
        # Type 2: M -> U_new, lambda*U+(1-lambda)*U -> M_new
        if type==2:
            #Next iteration of fixed point method
            U_new = HJ_solver(g, h, dt, N_t, mu_0, f_HJ, U_initial,np.copy(U),M,b,mu_1)
            U_new = U*lamda + (1-lamda)*U_new if i>1 else U_new
            M_new = FP_solver(g, M_terminal, np.copy(M), U_new, f_FP, b, h, dt, N_t)
            M_new = M*lamda + (1-lamda)*M_new if i>1 else M_new
            
            
            #Choose what times to calculate change and error
            if check_mid:
                start_index, stop_index = _middle_of_interval_indices(N_t,t_end)
            else:
                start_index, stop_index = 0,None
            
            #Calculate and track change in U, M
            U_change.append(_calc_rel_error(U[:,start_index:stop_index],
                                            U_new[:,start_index:stop_index],
                                            "infty"))
            M_change.append(_calc_rel_error(M[:,start_index:stop_index],
                                            M_new[:,start_index:stop_index],
                                            "K"))
            
            #Tracking approximation error
            if have_analytic:
                U_error.append(_calc_rel_error(U_new[:,-1],u_vec[:,-1],
                                            errortype="infty"))
                M_error.append(_calc_rel_error(M_new[:,0],m_vec[:,0],
                                            errortype="K"))
            
            #Status-plotting
            if i%plot_freq==plot_freq-1:
                _MFG_solver_plot_convergence_of_fixed_point_iterations(U_error,M_error,
                                                                    U_change, 
                                                                    M_change, lamda, 
                                                                    have_analytic)
            
            #Update U, M
            U, M = U_new, M_new

            #Stop loop if change in M and U are small
            if (U_change[-1] < stop) and (
                M_change[-1] < stop):
                print(f"Stopped due to very small change in resulting matrices.\niteration #{i+1}")
                break
    
    #Status-plotting
    _MFG_solver_plot_convergence_of_fixed_point_iterations(U_error, M_error,
                                                           U_change, M_change, 
                                                           lamda, have_analytic,
                                                           filename=filename)
    
    
    return U,M

