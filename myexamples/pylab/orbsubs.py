
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from numpy import linalg as LA
#from matplotlib import rc
#rc('text', usetex=True)
from numpy import polyfit
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from math import fmod
from scipy.signal import medfilt

from kepcart import *
from outils import * # useful short routines

angfac = 180.0/np.pi # for converting radians to degrees
twopi = np.pi*2.0

# read in an extended mass output file format fileroot_ext.txt
def readresfile(fileroot,ibody):
    filename = fileroot+'_ext'
    if (ibody > 0):
        filename = filename + '_{:0d}'.format(ibody)
    filename = filename + '.txt'
    print(filename)
    t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz,KErot,\
        PEspr,PEgrav,Etot,dEdtnow =\
        np.loadtxt(filename, skiprows=1, unpack='true')
    return t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz,\
        KErot,PEspr,PEgrav,Etot,dEdtnow


# read in numerical simulation output for two resolved bodies
# return spins, orbital elements and obliquities
# uses routines: crossprod_unit, dotprod, readresfile, keplerian
def read_two_bodies(simdir,froot,GM):
    fileroot=simdir+froot
    # format for ext files
    # t x y z vx vy vz omx omy omz llx lly llz Ixx Iyy Izz Ixy Iyz Ixz
    # KErot PEspr PEgrav Etot dEdtnow dEdtave
    # read in first resolved body
    t1  ,x1,y1,z1,vx1,vy1,vz1,omx1,omy1,omz1,\
        llx1,lly1,llz1,Ixx,Iyy,Izz,Ixy,Iyz,Ixz,KErot,PEspr,PEgrav,\
        Etot,dEdtnow = readresfile(fileroot,1)
    # read in second resolved body
    tarr,x2,y2,z2,vx2,vy2,vz2,omx2,omy2,omz2,\
        llx2,lly2,llz2,Ixx,Iyy,Izz,Ixy,Iyz,Ixz,KErot,PEspr,PEgrav,\
        Etot,dEdtnow = readresfile(fileroot,2)
    ns = len(t1) # number of timesteps
    # relative positions and velocities
    dx = x2-x1; dvx = vx2-vx1;
    dy = y2-y1; dvy = vy2-vy1;
    dz = z2-z1; dvz = vz2-vz1;
    aaarr = np.zeros(ns); eearr = np.zeros(ns); iiarr = np.zeros(ns)
    lnarr = np.zeros(ns); ararr = np.zeros(ns); maarr = np.zeros(ns)
    # compute the orbital elements
    for k in range(ns):
        aa,ee,ii,longnode,argperi,meananom=\
               keplerian(GM,dx[k],dy[k],dz[k],dvx[k],dvy[k],dvz[k])
        aaarr[k] = aa
        eearr[k] = ee
        iiarr[k] = ii
        lnarr[k] = longnode
        ararr[k] = argperi
        maarr[k] = meananom
    # total spin values
    om1 = np.sqrt(omx1**2 + omy1**2 + omz1**2)
    om2 = np.sqrt(omx2**2 + omy2**2 + omz2**2)
    # total angular momentums
    ll1 = np.sqrt(llx1**2 + lly1**2 + llz1**2)
    ll2 = np.sqrt(llx2**2 + lly2**2 + llz2**2)
    
    # compute orbit normal
    no_x,no_y,no_z=crossprod_unit(dx,dy,dz,dvx,dvy,dvz)
    # angular momentum unit vectors
    nlx1 = llx1/ll1; nly1 = lly1/ll1; nlz1 = llz1/ll1;
    nlx2 = llx2/ll2; nly2 = lly2/ll2; nlz2 = llz2/ll2;
    # compute obliquities
    ang_so1 = dotprod(nlx1,nly1,nlz1,no_x,no_y,no_z)
    ang_so1 = np.arccos(ang_so1)*angfac # obliquity body 1 in degrees
    ang_so2 = dotprod(nlx2,nly2,nlz2,no_x,no_y,no_z)
    ang_so2 = np.arccos(ang_so2)*angfac # obliquity body 2 in degrees
    obliquity_deg1 = ang_so1
    obliquity_deg2 = ang_so2
    
    #compute mean motion of binary
    meanmotion = np.sqrt(GM)/aaarr**1.5
    
    return tarr,aaarr,eearr,iiarr,lnarr,ararr,maarr,om1,om2,\
        obliquity_deg1,obliquity_deg2,meanmotion
    
    
