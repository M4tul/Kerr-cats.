# 2026 Michal Matulik (matulik@optics.upol.cz)
# 2026 Jan Provaznik (provaznik@optics.upol.cz)

import numpy as np
import numba as na
import scipy.special as ss
import scipy.optimize as so
import scipy.linalg as la
from scipy.linalg import expm
from scipy.optimize import minimize
import scipy.optimize as opt

#Computation of beta and norm 

def beta_func(beta,alpha,parity,theta):
    func = np.exp(-2*beta**2) * ( np.cos(theta - 4*beta*alpha) + np.exp(-2*alpha**2) ) \
    * 1/( 1+np.exp(-2*alpha**2) * np.cos(theta) ) 
    # func = np.exp(-2*beta**2)*(np.cos(theta - 4*beta*alpha))/((1+np.exp(-2*alpha**2)*np.cos(theta)))
    # func = np.exp(-2*beta**2)*(np.exp(-2*alpha**2) + np.sin(4*alpha*beta) )
    return func
    
def objective_of_minimize(beta, alpha, parity,theta):
    return -np.abs(beta_func(beta, alpha, parity,theta))
    # return -beta_func(beta, alpha, parity,theta)
    
def initial_condition_of_minimize(x,alpha,parity,theta):
    func = beta_func(x,alpha,parity,theta)
    idx_max = np.argmax(func)
    initial_cond = x[idx_max]
    return initial_cond

def compute_beta_norm(alpha,parity,theta):
    x= np.linspace(-np.pi,np.pi,2000)
    x0 = initial_condition_of_minimize(x,alpha,parity,theta)
    result = minimize(objective_of_minimize, x0, args=(alpha,parity,theta,), method= 'CG')
    beta = result.x[0]
    norm = beta_func(beta, alpha, parity,theta)
    return beta,norm

#Catability

def catability (op,rho):
    return np.trace(op@rho)

#Catability renormalized

def catability_norm (op,rho,eigenval,gamma):
    dim = rho.shape[0]
    I = np.eye(dim)
    I = eigenval*I
    tr1 = np.trace(rho @ op)
    tr2 = np.trace(rho @ I)
    return tr1 - tr2  
#Fidelity

def fidelity(op1,op2):
    return np.trace(op1 @ op2)
    

# Gaussian pure loss channel.

def apply_kraus (rho, ops):
    return np.einsum('kpq, qr, ksr -> ps', ops, rho, np.conj(ops))

def loss_channel_kraus (d, z):
    '''
    Constructs the Kraus representation of the Gaussian pure loss channel.

    Parameters
    ----------
    d : integer
        Dimension of the Hilbert space.
    z : np.floating
        Intensity transmittance of the channel.
    Returns
    -------
    np.ndarray with shape = (d, d, d)
        A stack of the Kraus operators representing the action of the channel.
    '''

    A = np.zeros(shape = (d, d, d), dtype = np.float64)
    for i in np.arange(d):
        k = np.arange(1, d - i)
        v = np.sqrt(np.arange(i + 1, d, dtype = np.float64).cumprod())
        A[k, i, i + k] = v
    A[0] = np.eye(d, dtype = np.float64)
    K = np.arange(d, dtype = np.float64)
    C = np.sqrt(1 - z) ** K / np.sqrt(ss.factorial(K))
    return \
        C[:, np.newaxis, np.newaxis] \
        * np.diag(np.sqrt(z) ** K)[np.newaxis, ...] \
        @ A

# Fock representation of the coherent state.
#

@na.njit(cache = True)
def coherent_state (dim, alpha):
    '''
    Constructs the coherent state in Fock basis on a
    truncated finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : integer
        Dimension of the Hilbert space restriction to construct the state on.
    alpha : np.complex
        Amplitude of the target coherent state.
    '''

    state = np.zeros(shape = dim, dtype = np.complex128)
    basis = np.arange(dim, dtype = np.float64)
    roots = np.sqrt(basis)

    state[0] = np.exp(- 0.5 * np.abs(alpha) ** 2)
    for k in np.arange(1, dim):
        state[k] = (alpha / roots[k]) * state[k - 1]
    return state


def cat_ideal(dim, alpha,theta):
    '''
    Constructs the cat state in Fock basis on a
    truncated finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : integer
        Dimension of the Hilbert space restriction to construct the state on.
    alpha : np.complex
        Amplitude of the target coherent state.
    theta : np.real
        Phase of the cat state, if +-np.pi/2 then kerr cat. 
    '''
    
    cat_state = 1/np.sqrt(2*(1 + np.cos(theta)*np.exp(-2*np.abs(alpha)**2)) )*(coherent_state(dim,alpha)\
              + np.exp(1j*theta)* coherent_state(dim,-alpha))
    cat_rho = np.outer(cat_state,np.conj(cat_state))
    return cat_rho

# Fock representation of the Gaussian squeezing unitary.
#
# Its elements are computed using the (Miatto, 2020) recurrent relation. The
# relation for squeezing is numerically stable. This is not true for Gaussian
# displacement (Provaznik, 2022).

@na.njit(cache = True)
def gaussian_squeezing (d, r):
    '''
    Squeezing operator constructed using the (Miatto, 2020) recurrent formula.

    Parameters
    ----------
    d : integer
        Dimension of the resulting matrix.
    r : np.floating
        Squeezing rate.

    Returns
    -------
    np.ndarray
        The resulting matrix of the squeezing operator.
    '''

    sech = 1.0 / np.cosh(r)
    tanh = np.tanh(r)
    sqrt = np.sqrt(np.arange(d + 1))

    O = np.zeros(shape = (d, d), dtype = np.float64)
    O[0, 0] = np.sqrt(sech)

    for m in range(d - 1):
        O[m + 1, 0] = - tanh * (sqrt[m] / sqrt[m + 1]) * O[m - 1, 0]
    for m in range(d):
        for n in range(d - 1):
            O[m, n + 1] = (
                sech * (sqrt[m] / sqrt[n + 1]) * O[m - 1, n] +
                tanh * (sqrt[n] / sqrt[n + 1]) * O[m, n - 1])
    return O

# Fock representation of the Gaussian displacement unitary.
#
# Its elements are computed using the (Cahill, 1969) explicit relation.

@na.njit
def _laguerre (N : int, a : int, x : complex) -> np.ndarray:
    '''
    Constructs a sequence of the first N Laguerre (associated) polynomials.
    Parameters
    ----------
    N : int
        Construct the first (0, 1, 2, N - 1) Laguerre polynomials.
    x : complex
        Value for which the polynomials are computed, as in L(n, a, x).
    Returns
    -------
    np.ndarray
        The numerical sequence.
    '''

    O = np.zeros(N, dtype = np.complex128)
    O[0] = 1.0
    for n in range(N - 1):
        O[n + 1] = ((2 * n + 1 + a - x) * O[n] - (n + a) * O[n - 1]) / (1 + n)
    return O
@na.njit
def _cumlog (d):
    F = np.arange(d)
    F[0] = 1
    return np.cumsum(np.log(F))
def gaussian_displacement (d, x) -> np.ndarray:
    '''
    Displacement operator constructed using the (Cahill, 1969) closed form
    formula, enhanced with exp-log trickery to deal with the factorials
    in the expression.
    Parameters
    ----------
    d : np.integer
        Dimension of the resulting matrix.
    x : np.complex
        Displacement amplitude.
    Returns
    -------
    np.ndarray with shape (d, d)
        The resulting matrix of the displacement operator.
    '''
    y = np.abs(x) ** 2
    
    # Correctly handles 0 magnitude by returning identity matrix
    if not(y > 0):
        return np.eye(d)

    # Handles negative and complex displacement amplitudes
    z = np.log(np.abs(x))
    q = np.angle(x)

    # Cumulative logarithm, look up table
    F = _cumlog(d)

    # Diagonal construction (lth diagonal at a time)
    O = np.zeros(shape = (d, d), dtype = np.complex128)
    for l in range(d):
        L = np.log(_laguerre(d - l, l, y))
        for n in range(d - l):
            m = n + l
            O[m, n] = np.exp(1j * q * l) * np.exp(l * z - 0.5 * y - 0.5 * (F[m] - F[n]) + L[n])
            O[n, m] = np.conj(O[m, n]) * (-1.0) ** l
    return O

