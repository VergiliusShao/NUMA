import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from skimage import io, color
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
def laplace_operator(M,N):
    I_M=sparse.eye(m=M,k=0)
    I_N=sparse.eye(m=N,k=0)
    D_M=-2.0*sparse.eye(m=M,k=0)+sparse.eye(m=M,k=-1)+sparse.eye(m=M,k=1)
    D_N=-2.0*sparse.eye(m=N,k=0)+sparse.eye(m=N,k=-1)+sparse.eye(m=N,k=1)
    delta=sparse.kron(I_M,D_N)+sparse.kron(D_M,I_N)
    return delta.tolil()




def _seamless_cloning(g,f,pos,tol=1e-1):
    #initialisation
    x0,y0=pos
    N_g,M_g=g.shape
    N_f,M_f=f.shape
    A=laplace_operator(M_g,N_g)
    g_vec=g.ravel(order='F')

    #set rhs
    b=A@g_vec
    
    #Update A_i, b_i with e_i^\top, h_i if i represent the value of a bounday pixel
   
    for i in range(N_g):
        #first column
        idx = i        
        A[idx,:]=0
        A[idx,idx]=1
        # get boundary value from f
        b[idx]= f[x0+i,y0]

        #last column
        idx = i + N_g*(M_g-1)        
        A[idx,:]=0
        A[idx,idx]=1
        # get boundary value from f
        b[idx]= f[x0+i,y0+M_g-1]

    for j in range(M_g):
        #first row of h
        idx = j*N_g        
        A[idx,:]=0
        A[idx,idx]=1
        # get boundary value from f
        b[idx]= f[x0,y0+j]
    
        #last row of h
        idx = N_g-1 + j*N_g        
        A[idx,:]=0
        A[idx,idx]=1
        # get boundary value from f
        b[idx]= f[x0+N_g-1,y0+j]

                
    #solve for h
    print('starting cg')
    h,_=linalg.cg(A=A,b=b,atol=1.0)
    print('cg accomplished')
    h=h.reshape(M_g,N_g, order='F')
    for i in range(N_g):
        for j in range(M_g):
            x = x0 + i
            y = y0 + j
            f[x,y]=h[i,j]
    return f

def seamless_cloning(g,f,pos,tol=1e-1):
    f_star=np.stack([_seamless_cloning(g[:,:,0],f[:,:,0],pos),_seamless_cloning(g[:,:,1],f[:,:,1],pos),_seamless_cloning(g[:,:,2],f[:,:,2],pos)],axis=2)
    return f_star



def seamless_cloning_mix(g,f,pos):
    return f

f =io.imread("./PA1/datasets/water.jpg")
g =io.imread("./PA1/datasets/bear.jpg")
f_star=seamless_cloning(g,f,(0,0))
plt.imshow(f_star)
plt.axis('off')  # optional: hide axes
plt.show()
