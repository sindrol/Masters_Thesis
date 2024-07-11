import numpy as np
from numpy import pi, sin, cos,exp,sqrt
import functions as fncs

class Test_HJ: # Parent class for easier construction of new test problems
    def __init__(self,N_x=50, t_end=2, b=1, lamda=None, r=None):
        self.N_x = N_x; self.b=b; self.t_end=t_end; self.lamda=lamda; self.r=r
        self.mu_1 = 1

        self.setup_kwargs = {"N_x":self.N_x, "lamda":self.lamda, "r":self.r,
                             "t_end":self.t_end, 
                            "b":self.b, "mu_1":self.mu_1, "u_0":self.u_0, 
                            "m_T":self.m_T, "mu_0_func":self.mu_0_func, 
                            "f_HJ_func":self.f_func, "f_FP_func":lambda x,t:0*x}

        self.setup_dict = fncs.setup(**self.setup_kwargs)
        self.setup_dict["M"]=self.m(self.setup_dict["grid_x"],
                                    self.setup_dict["grid_t"])
    
    def set_N_x(self, N_x):
        self.N_x = N_x
        self.setup_kwargs["N_x"]=self.N_x

        self.setup_dict = fncs.setup(**self.setup_kwargs)
        self.setup_dict["M"]=self.m(self.setup_dict["grid_x"],
                                    self.setup_dict["grid_t"])
        
    #Definition of test problem
    def u(self, x, t):
        return  None
    def m(self, x,t):
        return None
    def mu_0_func(self, x,t):
        return None

    #Depends on choice of u
    def u_t(self, x,t):
        return None
    def u_x(self, x,t):
        return None
    def u_xx(self, x,t):
        return None

    #The same every time
    def f_func(self, x,t):
        return -(self.u_t(x,t) + self.u_x(x,t)**2 - self.b*self.u_xx(x,t)+
                self.u(x,t)*( self.mu_0_func(x,t) + self.mu_1*self.m(x,t) ))
    def u_0(self, x):
        return self.u(x,0)
    def m_T(self,x):
        return self.m(x,self.t_end)

class Test1_HJ(Test_HJ):
    # Definition of test problem
    def u(self, x, t):
        return  cos(pi*x + 2*pi*t) + pi*(x-x**2)*sin(2*pi*t)
    def m(self, x,t):
        return 1-cos(3*pi*x)*cos(pi*t)
    def mu_0_func(self, x,t):
        return x*0+1

    # Partial derivatives of m and u
    def u_t(self, x,t):
        return -2*pi*sin(pi*x + 2*pi*t) + 2*pi**2*(x-x**2)*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x + 2*pi*t)+pi*(1-2*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x + 2*pi*t) - 2*pi*sin(2*pi*t)

class Test_FP: # Parent class for easier construction of new test problems
    def __init__(self,N_x=50, lamda=1, t_end=2, b=1):
        self.N_x = N_x; self.b=b; self.t_end=t_end; self.lamda=lamda

        self.setup_kwargs = {"N_x":self.N_x, "lamda":self.lamda, "t_end":self.t_end, 
                             "b":self.b, "u":self.u, "m_T":self.m_T,
                             "f_FP_func":self.f_func}

        self.setup_dict = fncs.setup_FP_test(**self.setup_kwargs)
    
    def set_N_x(self, N_x):
        self.N_x = N_x
        self.setup_kwargs["N_x"]=self.N_x

        self.setup_dict = fncs.setup_FP_test(**self.setup_kwargs)
        
    #Definition of test problem
    def m(self, x,t):
        return None
    def u(self, x, t):
        return  None

    #Depends on choice of m and u
    def m_t(self, x,t):
        return None
    def m_x(self, x,t):
        return None
    def m_xx(self, x,t):
        return None
    def u_x(self, x,t):
        return None
    def u_xx(self, x,t):
        return None
    
    #The same every time
    def div_term(self, x,t):
        return 2*(self.m_x(x,t)*self.u_x(x,t) + self.m(x,t)*self.u_xx(x,t))
    def f_func(self, x,t):
        return -self.m_t(x,t) -self.b*self.m_xx(x,t) -self.div_term(x,t)
    def u_0(self, x):
        return self.u(x,0)
    def m_T(self,x):
        return self.m(x,self.t_end)

class Test1_FP(Test_FP):
    # Definition of test problem
    def u(self, x, t):
        return  cos(pi*x + 2*pi*t) + pi*(x-x**2)*sin(2*pi*t)
    def m(self, x,t):
        return 1-cos(3*pi*x)*cos(pi*t)
    def mu_0_func(self, x,t):
        return x*0+1

    # Partial derivatives of m and u
    def m_t(self, x,t):
        return pi*cos(3*pi*x)*sin(pi*t)
    def m_x(self, x,t):
        return 3*pi*sin(3*pi*x) * cos(pi*t)
    def m_xx(self, x,t):
        return 9*pi**2*cos(3*pi*x) * cos(pi*t)
    
    def u_t(self, x,t):
        return -2*pi*sin(pi*x + 2*pi*t) + 2*pi**2*(x-x**2)*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x + 2*pi*t)+pi*(1-2*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x + 2*pi*t) - 2*pi*sin(2*pi*t)

class Test_MFG: # Parent class for easier construction of new test problems
    def __init__(self,N_x=50, lamda=1, t_end=2, b=1):
        self.N_x = N_x; self.b=b; self.t_end=t_end; self.lamda=lamda
        self.mu_1 = 1

        self.setup_kwargs = {"N_x":self.N_x, "lamda":self.lamda, "t_end":self.t_end, 
                                "b":self.b, "mu_1":self.mu_1, "u_0":self.u_0, 
                                "m_T":self.m_T, "mu_0_func":self.mu_0_func, 
                                "f_HJ_func":self.f_HJ_func, 
                                "f_FP_func":self.f_FP_func}

        self.setup_dict = fncs.setup(**self.setup_kwargs)
    
    def set_N_x(self, N_x):
        self.N_x = N_x
        self.setup_kwargs["N_x"]=self.N_x

        self.setup_dict = fncs.setup(**self.setup_kwargs)
        
    # Definition of test problem
    def m(self, x,t):
        return None
    def u(self, x, t):
        return  None
    def mu_0_func(self, x,t):
        return None

    # Depends on choice of m and u
    def m_t(self, x,t):
        return None
    def m_x(self, x,t):
        return None
    def m_xx(self, x,t):
        return None
    def u_t(self,x,t):
        return None
    def u_x(self, x,t):
        return None
    def u_xx(self, x,t):
        return None
    
    # The same every time
    def f_HJ_func(self, x,t):
        return -(self.u_t(x,t) + self.u_x(x,t)**2 - self.b*self.u_xx(x,t)+
                self.u(x,t)*( self.mu_0_func(x,t) + self.mu_1*self.m(x,t) ))
    def div_term(self, x,t):
        return 2*(self.m_x(x,t)*self.u_x(x,t) + self.m(x,t)*self.u_xx(x,t))
    def f_FP_func(self, x,t):
        return -self.m_t(x,t) -self.b*self.m_xx(x,t) -self.div_term(x,t)
    def u_0(self, x):
        return self.u(x,0)
    def m_T(self,x):
        return self.m(x,self.t_end)

class Test1_MFG(Test_MFG):
    # Definition of test problem
    def u(self, x, t):
        return  cos(pi*x + 2*pi*t) + pi*(x-x**2)*sin(2*pi*t)
    def m(self, x,t):
        return 1-cos(3*pi*x)*cos(pi*t)
    def mu_0_func(self, x,t):
        return x*0+1

    # Partial derivatives of m and u
    def m_t(self, x,t):
        return pi*cos(3*pi*x)*sin(pi*t)
    def m_x(self, x,t):
        return 3*pi*sin(3*pi*x) * cos(pi*t)
    def m_xx(self, x,t):
        return 9*pi**2*cos(3*pi*x) * cos(pi*t)
    
    def u_t(self, x,t):
        return -2*pi*sin(pi*x + 2*pi*t) + 2*pi**2*(x-x**2)*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x + 2*pi*t)+pi*(1-2*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x + 2*pi*t) - 2*pi*sin(2*pi*t)


# DVM problem with arguments as given by the PhD-thesis of Mazuryn ----------------------
class DVM:
    
    def __init__(self, N_x=100, lamda=1, t_end=10, gamma=0.02, I_0 = 1, K=1, T=24,
                 A=13, k_1=0.02, mort_scaling=100, mu_base = 10**-5, k_2 =0.02, 
                 x_clin=100, g_0=0.01, nu=10**-6, D=500, mu_1_star=1, sigma_sqr=20):
        
        #Normal model variables
        self.N_x = N_x; self.t_end=t_end; self.lamda=lamda
        #DVM variables
        self.gamma=gamma; self.I_0=I_0; self.K=K; self.A=A; self.k_1=k_1
        self.mort_scaling=mort_scaling; self.mu_base=mu_base; self.k_2=k_2
        self.x_clin=x_clin; self.g_0=g_0; self.nu=nu; self.T=T; self.D=D

        #Scaled quantities
        self.b=sigma_sqr/2 * T/D**2
        self.mu_1 = T/D * mu_1_star
        self.U_tilde = 2*nu*D**2/T
        
        self.setup_kwargs = {"N_x":self.N_x, "lamda":self.lamda, "t_end":self.t_end, 
                             "b":self.b, "mu_1":self.mu_1, "u_0":self.u_0, 
                             "m_T":self.m_T,"mu_0_func":self.mu_0_func, 
                             "f_HJ_func":self.f_func, "f_FP_func":lambda x,t:0*x}

        self.setup_dict = fncs.setup(**self.setup_kwargs)
        self.descaled_dict = {
            "x":self.setup_dict["x"]*self.D,
            "t":self.T*(self.t_end - self.setup_dict["t"]),
            "grid_x":self.setup_dict["grid_x"]*self.D,
            "grid_t":self.T*(self.t_end - self.setup_dict["grid_t"]),
            
        }
        
        self.U, self.M = self.setup_dict["U"], self.setup_dict["M"]
        self.U_descaled, self.M_descaled = None, None
        self.V_descaled = None
        
        self._descale_grid()
    
    #Set functions
    def set_N_x(self, N_x):
        self.N_x = N_x
        self.setup_kwargs["N_x"]=self.N_x
        self.setup_dict = fncs.setup(**self.setup_kwargs)
        
        self.set_U_M(self.setup_dict["U"], self.setup_dict["M"])
        self._descale_grid()
        
    def set_U_M(self,U,M):
        self.U=U
        self.M=M
        self._descale_U_M()
        self._descale_calculate_V()
    
    #DVM-functions
    def I_s(self,t):
        return self.I_0/(1+exp(self.A*cos(2*pi*t/self.T)))
    def I(self,x,t):
        return self.I_s(t)*exp(-self.k_1*x)
    def r(self,x,t):
        return 2/self.k_1*lambertw( (self.gamma * self.k_1 * sqrt(self.I(x,t)))/(
            2*sqrt(self.K + self.I(x,t)) ) ).real
    def mu_0_star(self,x,t):
        return self.mort_scaling*self.r(x,t)**2 + self.mu_base
    def harvest_rate_star(self, x,t):
        return self.g_0/(1+exp(self.k_2*(x-self.x_clin)))
    
    #Rescaled values & functions
    def f_func(self,x,t):
        return self.T/self.U_tilde * self.harvest_rate_star(self.D*x, self.T*(1-t))
    def mu_0_func(self,x,t):
        return self.T * self.mu_0_star(self.D*x, self.T*(1-t))
    
    #Self-chosen initial/terminal values
    def m_T(self,x):
        sigma=1/20
        return 1/(np.sqrt(2*pi)*sigma)*np.exp(-1/2 * (x-0.1)**2/sigma**2)
        ret = np.zeros(x.shape)
        for i in range(len(x)):
            if x[i]<2/5:
                ret[i] = 5/2 * (cos(5*pi/2 * x[i])+1)
            else:
                ret[i] = 0
        return ret
        #return 0*x+1
    def u_0(self,x):
        return 0*x-30
    
    
    #Descale functions
    def _descale_grid(self):
        self.grid_x_descaled = self.flip_matrix(self.D*self.setup_dict["grid_x"])
        self.grid_t_descaled = self.flip_matrix(self.T*(self.t_end - self.setup_dict["grid_t"]))
    
    def _descale_U_M(self):
        self.U_descaled = self.flip_matrix(-self.U_tilde * self.U)
        self.M_descaled = self.flip_matrix(1/self.D * self.M)
    
    def _descale_calculate_V(self):
        self.V_descaled = 1/self.nu * 1/(2*self.setup_dict["h"]*self.D) *(
            np.roll(self.U_descaled,-1)
            -np.roll(self.U_descaled,1))
    
    def flip_matrix(self, A):
        N_x,N_t = A.shape
        ret = np.copy(A)
        for i in range(N_t//2):
            for j in range(N_x):
                ret[j,i], ret[j,N_t-1-i] = ret[j,N_t-1-i], ret[j,i]
        return ret
            
# Other Test Systems --------------------------------------------------------------------
from scipy.special import erf
from scipy.special import lambertw

class Test2_HJ(Test_HJ):
    #Definition of test problem
    def u(self, x, t):
        return  -1+cos(pi*x)
    def m(self, x,t):
        return 1 + 0*x
    def mu_0_func(self, x,t):
        return 0*x

    #Depends on choice of u
    def u_t(self, x,t):
        return 0*x
    def u_x(self, x,t):
        return -pi*sin(pi*x)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x)
     
class Test3_HJ(Test_HJ):
    #Definition of test problem
    def u(self, x, t):
        return  -1+cos(pi*x)
    def m(self, x,t):
        return (exp(-1/2 * (x-1/2)**2 / (1/2 + 1/4 * sin(2*pi*t))**2) /
                (1/2 *sqrt(pi/2) * (sin(2*pi*t) + 2) * 
                 erf( sqrt(2) / (sin(2*pi*t)+2) )))
    def mu_0_func(self, x,t):
        return (1-x)*sin(pi*t)**2

    #Depends on choice of u
    def u_t(self, x,t):
        return 0*x
    def u_x(self, x,t):
        return -pi*sin(pi*x)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x)

class Test4_HJ(Test_HJ):
    #Definition of test problem
    def u(self, x, t):
        return  -1+cos(pi*x)+sin(2*pi*t)
    def m(self, x,t):
        return (exp(-1/2 * (x-1/2)**2 / (1/2 + 1/4 * sin(2*pi*t))**2) /
                (1/2 *sqrt(pi/2) * (sin(2*pi*t) + 2) * 
                 erf( sqrt(2) / (sin(2*pi*t)+2) )))
    def mu_0_func(self, x,t):
        return (1-x)*sin(pi*t)**2

    #Depends on choice of u
    def u_t(self, x,t):
        return 2*pi*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x)

class Test5_HJ(Test_HJ):
    #Definition of test problem
    def u(self, x, t):
        return  cos(pi*x + 2*pi*t) + pi*(x-x**2)*sin(2*pi*t)
    def m(self, x,t):
        return (exp(-1/2 * (x-1/2)**2 / (1/2 + 1/4 * sin(2*pi*t))**2) /
                (1/2 *sqrt(pi/2) * (sin(2*pi*t) + 2) * 
                 erf( sqrt(2) / (sin(2*pi*t)+2) )))
    def mu_0_func(self, x,t):
        return (1-x)*sin(pi*t)**2

    #Depends on choice of u
    def u_t(self, x,t):
        return -2*pi*sin(pi*x + 2*pi*t) + 2*pi**2*(x-x**2)*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x + 2*pi*t)+pi*(1-2*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x + 2*pi*t) - 2*pi*sin(2*pi*t)

class Test6_HJ(Test_HJ):
    #Definition of test problem
    a=1/6; sigma=1/10
    def u(self, x, t):
        return  cos(pi*x + 2*pi*t) + pi*(x-x**2)*sin(2*pi*t)
    def m(self, x,t):
        return 1/(np.sqrt(2*pi)*self.sigma)*np.exp(-1/2 * (x-1/2-self.a*sin(2*pi*t))**2/self.sigma**2)
    def mu_0_func(self, x,t):
        return x*0

    #Depends on choice of u
    def u_t(self, x,t):
        return -2*pi*sin(pi*x + 2*pi*t) + 2*pi**2*(x-x**2)*cos(2*pi*t)
    def u_x(self, x,t):
        return -pi*sin(pi*x + 2*pi*t)+pi*(1-2*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -pi**2*cos(pi*x + 2*pi*t) - 2*pi*sin(2*pi*t)

class Test7_HJ(Test_HJ):
    #Definition of test problem
    def u(self, x, t):
        return  cos(2*pi*x)*sin(2*pi*t)-1
    def m(self, x, t):
        return x*0
    def mu_0_func(self, x,t):
        return (1-x)**2 * cos(2*pi*t)**2+1

    #Depends on choice of u
    def u_t(self, x,t):
        return 2*pi*cos(2*pi*x)*cos(2*pi*t)
    def u_x(self, x,t):
        return -2*pi*sin(2*pi*x)*sin(2*pi*t)
    def u_xx(self, x,t):
        return -(2*pi)**2 *cos(2*pi*x)*sin(2*pi*t)

class Test2_FP(Test_FP):
    #Definition of test problem
    def m(self, x,t):
        return 1/self.t_end * ( t + (self.t_end-t)*2*x**2*(3-2*x) )
    def u(self, x, t):
        return  -1/(2*pi)*cos(2*pi*x)*(t+1)

    #Depends on choice of m and u
    def m_t(self, x,t):
        return 1/self.t_end * ( 1-2*x**2*(3-2*x) )
    def m_x(self, x,t):
        return 1/self.t_end * (self.t_end-t) * 12*x*(1-x)
    def m_xx(self, x,t):
        return 1/self.t_end * (self.t_end-t) * 12*(1-2*x)
    def u_x(self, x,t):
        return sin(2*pi*x)*(t+1)
    def u_xx(self, x,t):
        return 2*pi*cos(2*pi*x)*(t+1)

class Test3_FP(Test_FP):
    #Definition of test problem
    def m(self, x,t):
        return 1-cos(3*pi*x)*cos(pi*t)
    def u(self, x, t):
        return 5*t*x**2*(1/2 - 1/3 *x)

    #Depends on choice of m and u
    def m_t(self, x,t):
        return pi*cos(3*pi*x)*sin(pi*t)
    def m_x(self, x,t):
        return 3*pi*sin(3*pi*x) * cos(pi*t)
    def m_xx(self, x,t):
        return 9*pi**2*cos(3*pi*x) * cos(pi*t)
    def u_x(self, x,t):
        return 5*t*x*(1-x)
    def u_xx(self, x,t):
        return 5*t*(1-2*x)

class Test4_FP(Test_FP):
    #Definition of test problem
    a=1/6; sigma=1#/10
    def m(self, x,t):
        return 1/(np.sqrt(2*pi)*self.sigma)*np.exp(-1/2 * (x-1/2-self.a*sin(2*pi*t))**2/self.sigma**2)
    def _m_bar(self,x,t):
        return -(x-1/2-self.a*sin(2*pi*t))/self.sigma**2
    def _d(self,t):
        return -self.b/2 * self._m_bar(0,t)
    def _k(self,t):
        return -self.b/2 * self._m_bar(1,t)
    def u(self,x,t):
        return (self._k(t)-self._d(t))*1/2 *x**2 + self._d(t)*x

    #Depends on choice of m and u
    def m_x(self,x,t):
        return self._m_bar(x,t)*self.m(x,t)
    def m_xx(self,x,t):
        return (self._m_bar(x,t)**2-1/self.sigma**2)*self.m(x,t)
    def m_t(self,x,t):
        return -self.a*2*pi*cos(2*pi*t)*self._m_bar(x,t)*self.m(x,t)
    def u_x(self,x,t):
        return (self._k(t)-self._d(t))*x + self._d(t)
    def u_xx(self,x,t):
        return self._k(t)-self._d(t)

class Test2_MFG(Test_MFG):
    #Definition of test problem
    def m(self, x,t):
        return 1/self.t_end * ( t + (self.t_end-t)*2*x**2*(3-2*x) )
    def u(self, x, t):
        return  -1/(2*pi)*cos(2*pi*x)*(t+1)-1
    def mu_0_func(self, x,t):
        return (1-x)*sin(pi*t)**2

    #Depends on choice of m and u
    def m_t(self, x,t):
        return 1/self.t_end * ( 1-2*x**2*(3-2*x) )
    def m_x(self, x,t):
        return 1/self.t_end * (self.t_end-t) * 12*x*(1-x)
    def m_xx(self, x,t):
        return 1/self.t_end * (self.t_end-t) * 12*(1-2*x)
    def u_t(self,x,t):
        return -1/(2*pi)*cos(2*pi*x)
    def u_x(self, x,t):
        return sin(2*pi*x)*(t+1)
    def u_xx(self, x,t):
        return 2*pi*cos(2*pi*x)*(t+1)
 