import numpy as np
from scipy.signal import savgol_filter

class HySime(object):
    """ Hyperspectral signal subspace identification by minimum error. """

    def __init__(self):
        self.kf = None
        self.Ek = None

    def count(self, M):
        h, w, numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        Mr_filtered = savgol_filter(Mr, window_length=11, polyorder=3, axis=1)
        w, Rw = self.est_noise(Mr_filtered)
        self.kf, self.Ek = self.hysime(Mr_filtered, w, Rw)
        return self.kf, self.Ek
    
    
    def est_noise(self, y, noise_type='additive'):
        def est_additive_noise(r):
            small = 1e-6
            L, N = r.shape
            w=np.zeros((L,N), dtype=np.float64)
            RR=np.dot(r,r.T)
            RRi = np.linalg.pinv(RR+small*np.eye(L))
            RRi = np.matrix(RRi)
            for i in range(L):
                XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
                RRa = RR[:,i]
                RRa[i] = 0
                beta = np.dot(XX, RRa)
                beta[0,i]=0
                w[i,:] = r[i,:] - np.dot(beta,r)
            Rw = np.diag(np.diag(np.dot(w,w.T) / N))
            return w, Rw
        
        y = y.T
        L, N = y.shape
        #verb = 'poisson'
        if noise_type == 'poisson':
            sqy = np.sqrt(y * (y > 0))
            u, Ru = est_additive_noise(sqy)
            x = (sqy - u)**2
            w = np.sqrt(x)*u*2
            Rw = np.dot(w,w.T) / N
        # additive
        else:
            w, Rw = est_additive_noise(y)
        return w.T, Rw.T

    def hysime(self, y, n, Rn):
        y=y.T
        n=n.T
        Rn=Rn.T
        L, N = y.shape
        Ln, Nn = n.shape
        d1, d2 = Rn.shape

        x = y - n

        Ry = np.dot(y, y.T) / N
        Rx = np.dot(x, x.T) / N
        E, dx, V = np.linalg.svd(Rx)

        Rn = Rn+np.sum(np.diag(Rx))/L/10**5 * np.eye(L) # originally 10**5
        Py = np.diag(np.dot(E.T, np.dot(Ry,E)))
        Pn = np.diag(np.dot(E.T, np.dot(Rn,E)))
        cost_F = -Py + 2 * Pn
        kf = np.sum(cost_F < 0)
        ind_asc = np.argsort(cost_F)
        Ek = E[:, ind_asc[0:kf]]
        return kf, Ek # Ek.T ?