# A Python libaray for handling a variety of operations in continuous variable quantum inforomation theory with a focus on Gaussian states
# Vacuum contribution is normalized to unity. Key todo is to generalize this.
# Copyright 2021 Kieran Neil Wilkinson

from scipy import stats, special, linalg
from math import sqrt
import numpy as np

QS=np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
    ])
    
QS2=np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
    ])

class MType():
    HET  = 1
    HOMQ = 2
    HOMP = 3
    
class OpOrder():
    QQ = 1
    QP = 2

def Omga(n, opOrd=OpOrder.QP):
    """
    Compute the symplectic form

    :param n: Number of modes
    :opOrd : Order of the operators (OpOrder class)
    """ 

    if opOrd == OpOrder.QQ:
        return np.kron([[0, 1], [-1, 0]], np.eye(int(n)))
    
    return np.kron(np.eye(int(n)), [[0, 1], [-1, 0]])


######################################
# --     Common Gaussian CMs      -- #
######################################

def CMEPR(mu, k=1, err=0):
    """ Generate the CM of an EPR state

    Parameters
    ----------
    mu : float
        Variance of the EPR

    k : int 
        Sign of the quadratures

    err : float
        Error to introduce to the correlations
    """ 
    V = np.empty([4, 4])

    Z = np.eye(2)
    Z[1,1] = -1

    V[0:2, 0:2] = mu*np.eye(2)
    V[2:4, 0:2] = (1.0-err)*k*np.sqrt(mu*mu-1)*Z
    V[0:2, 2:4] = V[2:4, 0:2]
    V[2:4, 2:4] = V[0:2, 0:2]

    return V

########################################
# -  Covariance matrix operations   -- #
########################################

def Bs(T, n, m1, m2):
    """ 
    Compute the matrix for a beam splitter operation 
    
    :param T: transmissivity of the beam splitter
    :param N: number of modes of the global system  
    :param m1: first mode interacting with the beam splitter
    :param m2: second mode interacting with the beam splitter
    """

    Va = np.sqrt(T)
    Cova = np.sqrt(1-T)

    M = np.zeros([2*n, 2*n])

    for m in range(0, n):
        if m+1 == m1:
            M[m*2:m*2+2, m*2:m*2+2] = np.eye(2)*Va
            M[m*2:m*2+2, (m2-1)*2:(m2-1)*2+2] = np.eye(2)*Cova
            M[(m2-1)*2:(m2-1)*2+2, m*2:m*2+2] = -np.eye(2)*Cova
        elif m+1 == m2:
            M[m*2:m*2+2, m*2:m*2+2] = np.eye(2)*Va
        else:
            M[m*2:m*2+2, m*2:m*2+2] = np.eye(2)

    return M

def RMMDS(V, ms):
    """ 
    Remove modes from a covariance matrix

    :param V: input covariance matrix
    :param ms: list of mode indices to remove in any order
    """

    indices = []

    for md in ms:
        indices.append((md-1)*2)
        indices.append((md-1)*2 + 1)

    return np.delete(np.delete(V, indices, 1), indices, 0)

def RMX(x, ms):
    """ 
    Remove modes from a displacement vector

    :param x: input displacement vector
    :param ms: list of mode indices to remove in any order
    """
    indices = []

    for md in ms:
        indices.append((md-1)*2)
        indices.append((md-1)*2 + 1)

    return np.delete(x, indices)

##########################################
# --  Measurement of Gaussian states  -- #
##########################################

def MSRCM(V, m, tpe):
    """ 
    Compute the covariance matrix of a multimode state after measurement of a mode

    :param V: pre-measurement covariance matrix
    :param m: index of the mode to measure
    :param tpe: the measurement type (hom q, hom p or het) 
    """

    n = int(V.shape[0]/2)
    m = int(m)

    A = RMMDS(V, [m])
    B = V[2*(m-1):2*m, 2*(m-1):2*m] 
    
    C = np.empty([2, (n-1)*2])
    C[:, 0:2*(m-1)] = V[(m-1)*2:2*m, 0:2*(m-1)]
    C[:, 2*(m-1):2*(n-1)] = V[(m-1)*2:2*m, 2*m:2*n]

    if tpe == MType.HET:
        inv = np.linalg.pinv(B + np.eye(2))
        return A - C.T @ inv @ C
    else:
        Pi = np.zeros([2,2])

        if tpe == MType.HOMQ:
            Pi[0,0] = 1
        else:
            Pi[1,1] = 1

        prod = np.linalg.pinv(Pi @ B @ Pi)

        return A - C.T @ prod @ C

def MSRX(V, X, m, tpe, q, p):
    """ 
    Compute the mean value of a multimode state after measurement of a mode

    :param V: pre-measurement covariance matrix
    :param X: pre-measurement mean value vector
    :param m: index of the mode to measure
    :param tpe: the measurement type (hom q, hom p or het) 
    :param q: the outcome of the measurement of the q-quadrature
    :param p: the outcome of the measurement of the p-quadrature
    """

    n = int(V.shape[0]/2)

    XA = RMX(X, [m])
    B = V[2*(m-1):2*(m-1)+2, 2*(m-1):2*m]   
    
    C = np.empty([2, (n-1)*2])
    C[:, 0:2*(m-1)] = V[(m-1)*2:2*m, 0:2*(m-1)]
    C[:, 2*(m-1):2*(n-1)] = V[(m-1)*2:2*m, 2*m:2*n]

    d = X[2*(m-1):2*m] - [q, p]

    if tpe == MType.HET:
        inv = np.linalg.inv(B + np.eye(2))
        return XA - np.transpose(np.dot(np.dot(C.T, inv), d.T))
    else:
        Pi = np.zeros([2,2])

        if tpe == MType.HOMQ:
            Pi[0,0] = 1
        else:
            Pi[1,1] = 1

        prod = np.transpose(np.linalg.pinv(np.dot(np.dot(Pi, B), Pi)))

        return XA - np.transpose(np.dot(np.dot(C.T, prod.T), d.T))

def MSRP(V, X, m, tpe, q, p):
    """ 
    Compute the probability of measurement of a mode of a Gaussian quantum state

    :param V: pre-measurement covariance matrix
    :param X: pre-measurement mean value vector
    :param m: index of the mode to measure
    :param tpe: the measurement type (hom q, hom p or het) 
    :param q: the outcome of the measurement of the q-quadrature
    :param p: the outcome of the measurement of the p-quadrature
    """

    xB = X[2*(m-1):2*m]
    B = V[2*(m-1):2*m, 2*(m-1):2*m] 

    if tpe == MType.HET:
        d = xB - [q, p]
        norm = 2*np.pi*np.sqrt(np.linalg.det(B + np.eye(2)))
        return np.exp(-0.5*np.inner( np.dot(d, np.linalg.inv(B + np.eye(2))), d))/norm
    else:
        if tpe == MType.HOMQ:
            th = 0
            eta = q
        else:
            th = np.pi/2
            eta = p

        Rp = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        Rm = np.array([[np.cos(-th), np.sin(-th)], [-np.sin(-th), np.cos(-th)]])

        M22 = np.dot(Rp, np.dot(np.linalg.inv(B), Rm))
        ss = M22[1, 1]*np.linalg.det(B)

        return np.exp(-(eta - np.dot(Rp, xB)[0])**2/(2*ss))/np.sqrt(2*np.pi*ss)

def MSRVX(V, X, m, tpe, q, p):
    """ 
    Compute the mean value and covariance matrix of a multimode state after measurement of a mode

    :param V: pre-measurement covariance matrix
    :param X: pre-measurement mean value vector
    :param m: index of the mode to measure
    :param tpe: the measurement type (hom q, hom p or het) 
    :param q: the outcome of the measurement of the q-quadrature
    :param p: the outcome of the measurement of the p-quadrature
    """

    n = int(V.shape[0]/2)
    
    XA = RMX(X, [m])
    A = RMMDS(V, [m])
    B = V[2*m-2:2*m, 2*m-2:2*m] 
    
    C = np.empty([2, int((n-1)*2)])
    C[:, 0:2*(m-1)] = V[(m-1)*2:2*m, 0:2*(m-1)]
    C[:, 2*(m-1):2*(n-1)] = V[(m-1)*2:2*m, 2*m:2*n]

    d = X[2*(m-1):2*m] - [q, p]

    if tpe == MType.HET:
        inv = np.linalg.inv(B + np.eye(2))
    
        return (
            A - np.dot(np.dot(C.T, inv), C),
            XA - np.dot(np.dot(C.T, inv), d)
            )
    else:
        Pi = np.zeros([2,2])

        if tpe == MType.HOMQ:
            Pi[0,0] = 1
        else:
            Pi[1,1] = 1

        prod = np.transpose(np.linalg.pinv(np.dot(np.dot(Pi, B), Pi)))

        return (
            A - np.dot(np.dot(C.T, prod), C),
            XA - np.dot(np.dot(C.T, prod),d)
            )

def MMSRV(V, mds, tpes):
    """ 
    Compute the mean value of a multimode state after measurement of many modes

    :param V: pre-measurement covariance matrix
    :param mds: indices of the modes to measure
    :param tpes: the measurement types (hom q, hom p or het) 
    """

    for i in range(0, len(mds)):
        V = MSRCM(V, mds[i], tpes[i])

    return V

def MMSRVX(V, X, mds, tpes, mqs, mps):
    """ 
    Compute the mean value and covariance matrix of a multimode state after measurement of many modes

    :param V: pre-measurement covariance matrix
    :param X: pre-measurement mean value vector
    :param mds: indices of the modes to measure
    :param tpes: the measurement types (hom q, hom p or het) 
    :param qs: the outcomes of the measurements of the q-quadratures in order
    :param ps: the outcomes of the measurements of the p-quadratures in order
    """

    O = np.argsort(mds)
    mds = np.sort(mds)

    for i in range(0, len(mds)):
        V, X = MSRVX(V, X, mds[i]-i, tpes[O[i]], mqs[O[i]], mps[O[i]])

    return [V, X]

##########################################
# --   Operations of Gaussian states  -- #
##########################################

def gaussianFidelity(V1, V2, X1, X2):
    """
    Compute the fidelity between two Gaussian states

    :param V1: covariance matrix of state 1
    :param V2: covariance matrix of state 2
    :param X1: mean value of state 1
    :param X2: mean value of state 2
    """

    M = int(V1.shape[0]/2)
    Om = Omga(M, OpOrder.QP)

    VA = np.transpose(Om) @ np.linalg.inv(V1 + V2) @ (Om + V2 @ Om @ V1)
    inv = np.linalg.inv(VA @ Om)
    RtMat = linalg.sqrtm(np.identity(2*M) + (inv @ inv))
    Ft = np.linalg.det((RtMat + np.identity(2*M)) @ VA)
    d = X2-X1
    
    return (Ft/(linalg.det((V1+V2)/2)))**0.25*np.exp(-0.25*d.T @ np.linalg.inv(V1+V2) @ d)

def InfNorm(Xr, Vr, Xs, Vs):
    """
    Compute the inf norm between two Gaussian states \rho and \sigma
    
    K. P. Seshadreesan, L. Lami, and M. M. Wilde, Journal of Mathematical Physics 59, 072204 (2018), ISSN 00222488.

    :param Xr: mean value of \rho
    :param Vr: covariance matrix of \rho
    :param Xs: mean value of \sigma
    :param Vs: covariance matrix of \sigma
    """

    M = Vr.shape[0]/2
    om = Omga(M)

    dx = Xs-Xr

    rt1 = linalg.sqrtm(np.eye(2*M) + np.linalg.matrix_power(Vr @ om, -2))
    rt2 = linalg.sqrtm(np.eye(2*M) + np.linalg.matrix_power(om @ Vr, -2))

    V = Vr + rt1 @ Vr @ np.linalg.inv(Vs-Vr) @ Vr @ rt2

    nu = symEigs(V)

    exp =  np.exp(0.5*(dx @ np.linalg.inv(Vs-Vr) @ dx.T))

    return np.sqrt(np.linalg.det(Vs - 1j*om)/np.linalg.det(Vr - 1j*om))*np.sqrt((nu-1)/(nu+1))*exp

##########################################
# --       Entropy calculations       -- #
##########################################

def binEnt(p):
    """
    Compute the binary entropy for a given probability

    """
    if p < np.finfo(float).eps or p > 1-np.finfo(float).eps:
        return 0

    return stats.entropy([p, 1-p], base=2)

def h(v):
    """ 
    Computes entropy of a symplectic eigenvalue 
    
    :param v: symplectic eigenvalue
    """

    if v - 1.0 < np.finfo(float).eps:
        return 0

    return 0.5*(v + 1.0)*np.log2(0.5*(v + 1.0)) - 0.5*(v - 1.0)*np.log2(0.5*(v - 1.0))

def symEigs(V, opOrd=OpOrder.QP):
    """
    Compute the symplectic eigenvalues of a covariance matrix (CM)
    
    :param V: input covariance matrix 
    :param opOrd: Order of the quadrature operators in the CM
    """

    Om = Omga(V.shape[0]/2, opOrd=opOrd)

    eigsFull = np.linalg.eigvals(1j*Om @ V)
    eigsSort = np.sort(abs(eigsFull))
    eigs = np.delete(eigsSort, [x for x in range(0, V.shape[0], 2)])

    return eigs

def VNECM(V, opOrd=OpOrder.QP):
    """
    Find the entropy of a Gaussian quantum state from the covariance matrix

    :param V: input covariance matrix 
    :param opOrd: Order of the quadrature operators in the covariance matrix
    """

    eigs = symEigs(V, opOrd=opOrd)

    return np.sum([h(v) for v in eigs])
    
def VNErho(rho, base=2, lower=True):
    """
    Find the entropy of a state from the density matrix

    :param rho: density matrix of the state
    :param base: the base of the logarithm in the entropy calculation 
    :param lower: option to use the lower or upper triangle in diagonalization routine.

    """

    eigs = linalg.eigvalsh(rho, lower=lower)
    return stats.entropy(eigs[eigs>=0], base=base)


