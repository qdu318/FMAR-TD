import copy
import tensorly as tl
import scipy as sp
import numpy as np
from tensorly.decomposition import tucker
from .util.functions import svd_init, fit_ar


class FMAR(object):
    
    def __init__(self,ts, p, Rs, K, tol, seed=None, Ms_mode=4, verbose=0, convergence_loss=False):
        
        self._ts = ts
        self._ts_ori_shape = ts.shape
        self._N = len(ts.shape) - 1
        self.T = ts.shape[-1]
        self._p = p
        self._Rs = Rs
        self._K = K
        self._tol = tol
        self._Ms_mode = Ms_mode
        self._verbose = verbose
        self._convergence_loss = convergence_loss
        
        if seed is not None:
            np.random.seed()

    
    def _initilizer(self, T_hat, Js, Rs, Xs):

        M = [ np.random.random([j,r]) for j,r in zip( list(Js), Rs )]
        return M

    
    def _get_cores(self, Xs, Ms):
        s=[m.T for m in Ms]
        cores = [tl.tenalg.multi_mode_dot( x, s, modes=[i for i in range(len(Ms))] ) for x in Xs]
        return cores

    def _estimate_ar(self, cores, p):
        cores = copy.deepcopy(cores)
        alpha = fit_ar(cores, p)
        return alpha
    
    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _get_unfold_tensor(self, tensor, mode):
        
        if isinstance(tensor, list):
            return [ tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")   

    def _update_Ms(self, Ms, Xs, unfold_cores, n):

        T_hat = len(Xs)
        Len = len(Ms)
        begin_idx = self._p

        H = self._get_H(Ms, n)

        if self._Ms_mode == 1:
            if n< Len-1:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
            else:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(U_, V_)
        # orth in J1 J2
        elif self._Ms_mode == 2:
            if n< Len-1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)

        elif self._Ms_mode == 3:
            As = []
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(np.sum(As, axis=0))
            b = np.sum(Bs, axis=0)
            temp = np.dot(a, b)
            Ms[n] = temp / np.linalg.norm(temp)
        # all orth
        elif self._Ms_mode == 4:
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            b = np.sum(Bs, axis=0)
            U_, _, V_ = np.linalg.svd(b, full_matrices=False)
            Ms[n] = np.dot(U_, V_)
        # only orth in J1
        elif self._Ms_mode == 5:
            if n==0:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
        # only orth in J2
        elif self._Ms_mode == 6:
            if n==1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
        return Ms


    def _compute_convergence(self, new_M, old_M):
        
        new_old = [ n-o for n, o in zip(new_M, old_M)]
        
        a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_M], axis=0)
        return a/b

    def _update_cores(self, n, Ms, Xs,  cores, alpha, lam=1):

        begin_idx = self._p
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Ms, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            a = np.sum([ alpha[i] * self._get_unfold_tensor(cores[t-(i+1)], n) for i in range(self._p)], axis=0 )
            unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Ms[n].T, unfold_Xs), H.T) + a)
        return unfold_cores

    def update_cores2(self,n, Ms, Xs, cores,B):
        begin_idx = 4
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Ms, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs =np.array(self._get_unfold_tensor(Xs[t], n))
            cores_np=np.array(cores)
            pred,_=self.mar(cores_np[:t -1,:,:], B, maxiter=1)
            a = pred[:,:,-1]
            unfold_cores[t] = 1 / 2 * ( np.dot(np.dot(Ms[n].T, unfold_Xs), H.T) + a)
        return unfold_cores

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs

    def _get_H(self, Ms, n):
        ab = Ms[::-1]
        Hs = tl.tenalg.kronecker([m.T for m, i in zip(Ms[::-1], reversed(range(len(Ms)))) if i!= n ])
        return Hs

    def mar(self,X,  B, maxiter=1):
        X=np.moveaxis(X,0,2)
        m, n, T = X.shape
        pred_step=1
        for it in range(maxiter):
            temp0 = B.T @ B
            temp1 = np.zeros((m, m))
            temp2 = np.zeros((m, m))
            for t in range(1, T):
                temp1 += X[:, :, t] @ B @ X[:, :, t - 1].T
                temp2 += X[:, :, t - 1] @ temp0 @ X[:, :, t - 1].T
            A = temp1 @ np.linalg.inv(temp2)
            temp0 = A.T @ A
            temp1 = np.zeros((n, n))
            temp2 = np.zeros((n, n))
            for t in range(1, T):
                temp1 += X[:, :, t].T @ A @ X[:, :, t - 1]
                temp2 += X[:, :, t - 1].T @ temp0 @ X[:, :, t - 1]
            B = temp1 @ np.linalg.inv(temp2)
        tensor = np.append(X, np.zeros((m, n, pred_step)), axis=2)
        for s in range(pred_step):
            tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
        return tensor[:, :, - pred_step:],  B

    def run(self):
        
        result, loss = self._run()
        
        if self._convergence_loss:
            
            return result, loss            
        
        return result, None

    def _run(self):

        Xs = self._get_Xs(self._ts)

        con_loss = []

        Ms = self._initilizer(len(Xs), Xs[0].shape, self._Rs, Xs)
        B = np.random.randn(self._Rs[1], self._Rs[1])
        for k in range(self._K):
            print(f"epoch={k}\n")
            old_Ms = Ms.copy()
            # get cores
            cores = self._get_cores(Xs, Ms)
            cores_np=np.array(cores)
            _,B = self.mar(cores_np, B)

            for n in range(len(self._Rs)):    #mode n
                cores_shape = cores[0].shape
                unfold_cores = self.update_cores2(n, Ms, Xs, cores, B)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                Ms = self._update_Ms(Ms, Xs, unfold_cores, n)

            # convergence check
            convergence = self._compute_convergence(Ms, old_Ms)
            con_loss.append(convergence)
            
            if k%10 == 0:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))

            if self._tol > convergence:
                if self._verbose == 1: 
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))

                break

   
        # Forecasting
        cores = self._get_cores(Xs, Ms)

        pred, _ = self.mar(cores_np[:, :, :], B, maxiter=1)
        pred=pred.reshape(5,5)
        new_X = tl.tenalg.multi_mode_dot(pred, Ms)
        print(new_X)
        Xs.append(new_X)
        return Xs, con_loss
