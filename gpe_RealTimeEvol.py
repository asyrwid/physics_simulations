import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict
import scipy.optimize 
import mpl_toolkits.axisartist as axisartist
import os
import math
import cmath
import scipy.optimize as opt
import random
import operator
import copy
import sys
import scipy.integrate as integrate
import scipy.sparse as sp
from numpy.linalg import inv
from scipy import sparse
from timeit import default_timer as timer
from scipy.integrate import dblquad
import numba
from numba import jit
import pylab as pl
from IPython import display
#%matplotlib inline
#from matplotlib import rc
#rc('text', usetex=True)
#mpl.rc('font',family='Times New Roman')
 

# ====================== Matrices generation ============================
def Scipylaplacian1D(N,d):
    diag = np.ones([N])
    mat = sp.spdiags([diag, diag,-2*diag,diag, diag],[-N+1,-1,0,1,N-1],N,N)
    M0 = np.array(mat.toarray(),dtype = np.complex128)
    return 1./4*M0/(d*d)
def InvMatrix1(N, w, d, dt): # inverse of O1 matrix generation
    M0 = Scipylaplacian1D(N,d)
    N0 = (N-1.0)/2.
    for i in range(0,N):
        x2 = (d*(i-N0))**2
        add = -1/4.*w**2*( x2 ) + 1j*1./dt
        M0[i,i] += add
    return inv(M0)
def Matrix2(N, w, d, dt): # O2 matrix generation
    M0 = -Scipylaplacian1D(N,d)
    N0 = (N-1.0)/2.
    for i in range(0,N):
        x2 = (d*(i-N0))**2
        add = 1/4.*w**2*( x2 ) + 1j/dt
        M0[i,i] += add
    return M0
# ============  NONLINEARITY =======================
@jit(nopython=True)
def nonl2(fam, fbm, gpar, gA, gB, d, gAB):
    gg = np.absolute(gpar)
    m=len(fbm)
    
    dax = 0.
    dbx = 0.
    lapax = 0.
    lapbx = 0.
    
    fan = np.zeros((m), dtype=np.complex128)
    fbn = np.zeros((m), dtype=np.complex128)
    for i in range(0,m):
            
        ip = i+1
        im = i-1         
        if i == 0:
            im = m-1              
        if i == m-1:
            ip = 0
            
        dax = 1./(2.*d)*(fam[ip] - fam[im])
        dbx = 1./(2.*d)*(fbm[ip] - fbm[im])          
        lapax = 1./(d*d)*( -2.*fam[i] + fam[ip] + fam[im] )
        lapbx = 1./(d*d)*( -2.*fbm[i] + fbm[ip] + fbm[im] )
        jax = 1./(2.*1j)*(np.conjugate(fam[i])*dax - fam[i]*np.conjugate(dax) )
        jbx = 1./(2.*1j)*(np.conjugate(fbm[i])*dbx - fbm[i]*np.conjugate(dbx) )            
        # regularizing terms related to ja.ja & jb.jb
        jja1 = (2./1j)*(jax*dax)
        jja2 = (-1./2.)*fam[i]*(np.conjugate(fam[i])*lapax - fam[i]*np.conjugate(lapax) )
        jjb1 = (2./1j)*(jbx*dbx)
        jjb2 = (-1./2.)*fbm[i]*(np.conjugate(fbm[i])*lapbx - fbm[i]*np.conjugate(lapbx) )
        jaja = gg/2.*(jja1 + jja2)
        jbjb = gg/2.*(jjb1 + jjb2)
        # parallel drag
        jABa1 = (1./1j)*(jbx*dax)
        jABa2 = -(1./4.)*fam[i]*(np.conjugate(fbm[i])*lapbx - fbm[i]*np.conjugate(lapbx) )
        jABb1 = (1./1j)*(jax*dbx)
        jABb2 = -(1./4.)*fbm[i]*(np.conjugate(fam[i])*lapax - fam[i]*np.conjugate(lapax) )
        jABa = gpar*(jABa1 + jABa2)
        jABb = gpar*(jABb1 + jABb2)
            
        fan[i] = jaja + jABa + gA*fam[i]*np.absolute(fam[i])**2 + gAB*fam[i]*np.absolute(fbm[i])**2
        fbn[i] = jbjb + jABb + gB*fbm[i]*np.absolute(fbm[i])**2 + gAB*fbm[i]*np.absolute(fam[i])**2
    return fan, fbn
# ============  Cranck proceurde =======================
def cranck(Tab_fa,Tab_fb,fa0,fb0,Nsteps,Nsaved,N,wa,wb,gpar,d,dt,Oa,Ob,invOa,invOb,gA,gB,gAB,PAS,PBS,ES):
    # Oa and Ob must be sparse !
    N0 = (N-1.0)/2.
    gg = np.absolute(gpar)
    norm_a0 = (np.sum(np.absolute(fa0)**2)*d)**(-1./2) 
    norm_b0 = (np.sum(np.absolute(fb0)**2)*d)**(-1./2) 
    fa0 = (fa0*norm_a0)
    fb0 = (fb0*norm_b0)
    fa1 = fa0
    fb1 = fb0    
    fam =(3.*fa1 - fa0)/2.
    fbm =(3.*fb1 - fb0)/2.
    start0 = timer()
    
    for sav in range(1,Nsaved+1): 
        for step in range(0, Nsteps):
            # caclutate linear part: O2 matrix action
            fa12 = Oa.dot(fa1)
            fb12 = Ob.dot(fb1)
            #calculate nonlinearities, 
            fan, fbn =  nonl2(fam, fbm, gpar, gA, gB, d, gAB)
                       
            fa12 = fa12 + fan
            fb12 = fb12 + fbn
            # calculate final vector: inverse of O1 action
            fa2 = invOa.dot(fa12)
            fb2 = invOb.dot(fb12)
            
            if np.real(dt) < pow(10.,-10.):
                norm_a = (np.sum(np.absolute(fa2)**2)*d)**(-1./2) 
                fa2 = fa2*norm_a
                norm_b = (np.sum(np.absolute(fb2)**2)*d)**(-1./2) 
                fb2 = fb2*norm_b
                                
            fa0 = fa1
            fb0 = fb1
            fa1 = fa2
            fb1 = fb2
            fam = (3*fa1 - fa0)/2.
            fbm = (3*fb1 - fb0)/2.
                
        fa3 = fa1
        fb3 = fb1
        Tab_fa[sav] = fa3
        Tab_fb[sav] = fb3
      
        n0 = (N-1.0)/2.
        en = 0.
        Fa2 = fa3
        Fb2 = fb3
        mm=len(Fa2)
        pa=0.
        pb=0.
        
        for i in range(0,mm):
            ip = i+1
            im = i-1
            if i == 0:
                im = mm-1
            if i == mm-1:
                ip = 0

            lap_a = -1./2.*np.conjugate(Fa2[i])*1./(d*d)*(Fa2[ip]+Fa2[im]-2*Fa2[i])
            lap_b = -1./2.*np.conjugate(Fb2[i])*1./(d*d)*(Fb2[ip]+Fb2[im]-2*Fb2[i])      
            dax = 1./(2.*d)*(Fa2[ip] - Fa2[im])
            dbx = 1./(2.*d)*(Fb2[ip] - Fb2[im])
            x2 = (d*(i-n0))**2
            harm = 1./2.*(wa**2)*x2*(np.absolute(Fa2[i]) )**2+1./2.*(wb**2)*x2*(np.absolute(Fb2[i]) )**2
            jax = 1./(2.*1j)*(np.conjugate(Fa2[i])*dax - Fa2[i]*np.conjugate(dax) )
            jbx = 1./(2.*1j)*(np.conjugate(Fb2[i])*dbx - Fb2[i]*np.conjugate(dbx) )
            jj = 1./2.*gg*( (jax**2) +(jbx**2) )
            jjab = gpar*jax*jbx 
            EAB = gAB*((np.absolute(Fa2[i]))**2)*((np.absolute(Fb2[i]))**2) 
                
            en += lap_a+lap_b+harm+jj+jjab + ( gA*np.absolute(Fa2[i])**4 + gB*np.absolute(Fb2[i])**4 )/2.+ EAB
            pa += -1j*np.conjugate(Fa2[i])*dax*d
            pb += -1j*np.conjugate(Fb2[i])*dbx*d
            
        norm_A = (np.sum(np.absolute(Fa2)**2)*d)
        norm_B = (np.sum(np.absolute(Fb2)**2)*d)
        PAS.append(pa)
        PBS.append(pb)
        ES.append(en)
        end0 = timer()
        tt = (end0 - start0)/60./sav
        left = Nsaved-sav
        ET = tt*left

        print([sav,"e=",np.round(np.real(en)*d,8),"n=",np.round(norm_A,8), np.round(norm_B,8),
           "p=", np.round(pa,5),np.round(pb,5), " ET = ",np.round(ET,3)],flush=True )

# = = = = = RealTimeEvolution!  = = = = = 
# load initial states
import scipy.io
ta = "/home/asyrwid/vecdrag/bright_black_1D_PBC/dat/Initial_Black_N=300_L=1_g=20_v=pibyl_centered.mat"
tb = "/home/asyrwid/vecdrag/bright_black_1D_PBC/dat_initial_bright/Initial_Bright_N=300_L=1_g=-20_v=0_centered.mat"

A = scipy.io.loadmat(ta)['out']
B = scipy.io.loadmat(tb)['out']

# discretization
L=1.
N = 300; #  number of points along x direction
n0 =(N-1.)/2.
d = L/N

# modify initial states if needed
#faa = A[len(A)-1]
fbb = B[len(B)-1] 
#faa = np.roll(faa,0,axis = 0)
#fbb = np.roll(fbb,0,axis = 0)
#faa = faa*np.exp(1j*np.arange(0.,2.*np.pi,2.*np.pi/N))
faa = np.ones((N),dtype=np.complex128)
fbb = fbb*np.exp(1j*np.arange(0,2*np.pi,2*np.pi/N))

#for i in range(0,len(faa)):
	#faa[i] = faa[i]*np.exp(0*1j*d*i) 
	#fbb[i] = fbb[i]*np.exp(0*1j*d*i)

print("state prepared", flush=True)

# directory and file names
f0="/home/asyrwid/vecdrag/bright_black_1D_PBC/dat/"
tn  ="N=300_"
tl  ="L=1_"
tpar="gpar=0_1"
tga ="ga=20_"
tgb ="gb=-20_"
tgab="gab=0_"
tvs ="va=2pi_vb=0_"
tdts="dt=0_0000001_Dt=0_01.mat"

filea = f0+"a_"+tn+tl+tpar+tga+tgb+tgab+tvs+tdts
fileb = f0+"b_"+tn+tl+tpar+tga+tgb+tgab+tvs+tdts
PFa   = f0+"P_a_"+tn+tl+tpar+tga+tgb+tgab+tvs+tdts
PFb   = f0+"P_b_"+tn+tl+tpar+tga+tgb+tgab+tvs+tdts
Efs   = f0+"Energy_"+tn+tl+tpar+tga+tgb+tgab+tvs+tdts

dt = 0.0000001
# trap parameters (assume that masses ma = mb = 1)
wa = 0;
wb = 0;

# phase transition for g= -pi^2
gA =  20
gB = -20
gAB = 20.
gpar = 0.1

# allocate memory
Nsaved = 2000
# Number of dt steps between consecutive outcomes (so total number of steps = Nsteps*Nsaved)
Nsteps = 100000

Tab_fa2 = np.zeros((Nsaved+1,N),dtype = np.complex128)
Tab_fb2 = np.zeros((Nsaved+1,N),dtype = np.complex128)

Tab_fa2[0] = faa
Tab_fb2[0] = fbb

Oa = sparse.csr_matrix( Matrix2(N, wa, d, dt) )
Ob = sparse.csr_matrix( Matrix2(N, wb, d, dt) )
invOa = InvMatrix1(N, wa, d, dt)
invOb = InvMatrix1(N, wb, d, dt)

PAS=[]
PBS=[]
ES=[]

start = timer()
cranck(Tab_fa2, Tab_fb2, faa, fbb, Nsteps, Nsaved, N, wa,wb,gpar,d,dt,Oa,Ob,invOa,invOb, gA, gB, gAB, PAS,PBS,ES)
end = timer()
print("\n time = ")
print(end - start)
print("\n")

import scipy.io
scipy.io.savemat(filea, mdict={'out': Tab_fa2}, oned_as='row')
scipy.io.savemat(fileb, mdict={'out': Tab_fb2}, oned_as='row')

scipy.io.savemat(PFa, mdict={'out': PAS}, oned_as='row')
scipy.io.savemat(PFb, mdict={'out': PBS}, oned_as='row')
scipy.io.savemat(Efs, mdict={'out': ES}, oned_as='row')
