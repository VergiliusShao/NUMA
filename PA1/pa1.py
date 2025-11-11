import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from skimage import io, color
import matplotlib.pyplot as plt
def laplace_operator(M,N):
    I_M=sparse.eye(m=M,k=0)
    I_N=sparse.eye(m=N,k=0)
    D_M=-2.0*sparse.eye(m=M,k=0)+sparse.eye(m=M,k=-1)+sparse.eye(m=M,k=1)
    D_N=-2.0*sparse.eye(m=N,k=0)+sparse.eye(m=N,k=-1)+sparse.eye(m=N,k=1)
    delta=sparse.kron(I_M,D_N)+sparse.kron(D_M,I_N)
    return delta




def _seamless_cloning(g,f,pos,tol=1e-1):
    M_g,N_g=g.shape
    M_f,N_f=f.shape
    delta=laplace_operator(M_g,N_g)
    g_vec=g.reshape(-1,1)
    #solve h
    delta_g=np.ravel((delta@g_vec))
    h,_=linalg.cg(A=delta,b=delta_g,atol=1e-1)
    h=h.reshape(M_g,N_g)

    #place g to f
    x0,y0=pos
    for i in range(M_g):
        for j in range(N_g):
            x = x0 + i
            y = y0 + j

        # Skip boundary entries of f
            if not (x == 0 or x >= M_f-1 or y == 0 or y >= N_f-1):
                f[x, y] = h[i,j]
    return f

def seamless_cloning(g,f,pos,tol=1e-1):
    f_star=np.stack([_seamless_cloning(g[:,:,0],f[:,:,0],pos),_seamless_cloning(g[:,:,1],f[:,:,1],pos),_seamless_cloning(g[:,:,2],f[:,:,2],pos)],axis=2)
    return f_star



def seamless_cloning_mix(g,f,pos):
    return f


f =io.imread("./PA1/bird.jpg")
g =io.imread("./PA1/plane.jpg")
f_star=seamless_cloning(g,f,(0,500))
plt.imshow(f_star)
plt.axis('off')  # optional: hide axes
plt.show()