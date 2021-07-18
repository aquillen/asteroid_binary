
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
from scipy.signal import savgol_filter


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
        llx1,lly1,llz1,Ixx1,Iyy1,Izz1,Ixy1,Iyz1,Ixz1,KErot1,PEspr1,PEgrav1,\
        Etot,dEdtnow = readresfile(fileroot,1)
    # read in second resolved body
    t2  ,x2,y2,z2,vx2,vy2,vz2,omx2,omy2,omz2,\
        llx2,lly2,llz2,Ixx2,Iyy2,Izz2,Ixy2,Iyz2,Ixz2,KErot2,PEspr2,PEgrav2,\
        Etot,dEdtnow = readresfile(fileroot,2)
    ns1 = len(t1) # number of timesteps
    ns2 = len(t2)
    ns = min(ns1,ns2)
    tarr = t2[0:ns]
    # relative positions and velocities
    dx = x2[0:ns]-x1[0:ns]; dvx = vx2[0:ns]-vx1[0:ns];
    dy = y2[0:ns]-y1[0:ns]; dvy = vy2[0:ns]-vy1[0:ns];
    dz = z2[0:ns]-z1[0:ns]; dvz = vz2[0:ns]-vz1[0:ns];
    r_B = np.sqrt(dx**2 + dy**2 + dz**2)   # distance between bodies
    # angle of orbit on xy plane
    theta_B = np.arctan2(dy,dx)
    dx_hat = dx/r_B # normalized direction between primary and secondary
    dy_hat = dy/r_B
    dz_hat = dz/r_B
    
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
    om1 = np.sqrt(omx1[0:ns]**2 + omy1[0:ns]**2 + omz1[0:ns]**2)
    om2 = np.sqrt(omx2[0:ns]**2 + omy2[0:ns]**2 + omz2[0:ns]**2)

    # total angular momentums
    ll1 = np.sqrt(llx1[0:ns]**2 + lly1[0:ns]**2 + llz1[0:ns]**2)
    ll2 = np.sqrt(llx2[0:ns]**2 + lly2[0:ns]**2 + llz2[0:ns]**2)
    
    # compute orbit normal
    no_x,no_y,no_z=crossprod_unit(dx,dy,dz,dvx,dvy,dvz)
    # angular momentum unit vectors
    nlx1 = llx1[0:ns]/ll1; nly1 = lly1[0:ns]/ll1; nlz1 = llz1[0:ns]/ll1;
    nlx2 = llx2[0:ns]/ll2; nly2 = lly2[0:ns]/ll2; nlz2 = llz2[0:ns]/ll2;
    # spin unit vectors
    nomx1 = omx1[0:ns]/om1;  nomy1 = omy1[0:ns]/om1;  nomz1 = omz1[0:ns]/om1;
    nomx2 = omx2[0:ns]/om2;  nomy2 = omy2[0:ns]/om2;  nomz2 = omz2[0:ns]/om2;
    # compute obliquities, angle between orbit normal and body angular momentum
    ang_lo1 = dotprod(nlx1,nly1,nlz1,no_x,no_y,no_z)
    ang_lo1 = np.arccos(ang_lo1) # obliquity body 1
    ang_lo2 = dotprod(nlx2,nly2,nlz2,no_x,no_y,no_z)
    ang_lo2 = np.arccos(ang_lo2) # obliquity body 2
    # compute obliquities, angle between orbit normal and body spin
    ang_so1 = dotprod(nomx1,nomy1,nomz1,no_x,no_y,no_z)
    ang_so1 = np.arccos(ang_so1) # obliquity body 1
    ang_so2 = dotprod(nomx2,nomy2,nomz2,no_x,no_y,no_z)
    ang_so2 = np.arccos(ang_so2) # obliquity body 2
    
    lobliquity_deg1 = ang_lo1*angfac
    lobliquity_deg2 = ang_lo2*angfac
    sobliquity_deg1 = ang_so1*angfac
    sobliquity_deg2 = ang_so2*angfac
    
    # compute angle between body angular momentum and spin angular momentum
    #Jang1 = dotprod(nlx1,nly1,nlz1,omx1,omy1,omz1)/om1
    #Jang1_deg = np.arccos(Jang1)*angfac
    #Jang2 = dotprod(nlx2,nly2,nlz2,omx2,omy2,omz2)/om2
    #Jang2_deg = np.arccos(Jang2)*angfac
    
    # spin angular momentum of body1,2 projected onto xy plane
    lprec_ang1 = np.arctan2(nly1,nlx1)
    lprec_ang2 = np.arctan2(nly2,nlx2)
    # spin body 1,2 projected onto xy plane
    sprec_ang1 = np.arctan2(nomy1,nomx1)
    sprec_ang2 = np.arctan2(nomy2,nomx2)
    
    #compute mean motion of binary
    meanmotion = np.sqrt(GM)/aaarr**1.5
    mm_smo = meanmotion
    r_smo = r_B
    if (len(r_smo)>500):
        r_smo  = savgol_filter(r_smo,  201, 2, mode='nearest')
        mm_smo = savgol_filter(mm_smo, 201, 2, mode='nearest')
        
    delta_r = r_B - r_smo  # radial deviations from local smoothed value
    v_r = (dvx*dx + dvy*dy + dvz*dz)/r_B # radial velocity
    theta_k = np.arctan2(delta_r*mm_smo,v_r)  # epicyclic angle
    
    # compute a libration angle
    # compute the angle between the body principal major axis,
    #    projected onto the orbital plane
    #    and the direction to the binary
    # dx,dy,dz is to secondary from primary
    
    a_x,a_y,a_z = crossprod_unit(no_x,no_y,no_z,dx_hat,dy_hat,dz_hat)
    # this is a vector in orbit plane that is perpendicular to orbit
    # normal and perpendicular to dr_hat direction between primary and secondary
    
    #xhat_x = om1*0.0 + 1.0   #  a vector of ones
    #xhat_y = om1*0.0   #  a vector of zeros
    #xhat_z = om1*0.0   #  a vector of zeros
    #oyhat_x,oyhat_y,oyhat_z =crossprod_unit(no_x,no_y,no_z,xhat_x,xhat_y,xhat_z)
    # a unit vector in orbital plane that is perpendicular to (1,0,0)
    #oxhat_x,oxhat_y,oxhat_z =crossprod_unit(no_x,no_y,no_z,oyhat_x,oyhat_y,oyhat_z)
    # a unit vector in orbital plane that is perpendicular to oyhat
    
    lib_angle2 = om1*0.0 # to store the libration angle
    phi1 = om1*0.0 # to store orientation angle of primary
    phi2 = om1*0.0 # to store orientation angle of secondary
    theta1_NPA = om1*0.0  # to store NPA angle for body 1
    theta2_NPA = om1*0.0  # to store NPA angle for body 2
    nprec_ang2= om1*0.0  #  to store short axis angle on xy plane
    ntilt2 = om1*0  # store angle between short axis and orbit normal
    longtilt2 = om1*0  # store angle between long axis and orbit normal
    for k in range(ns):
        # vmin corresponds to long axis of body, vmax to shortest axis
        # eigenvectors are body orientation axes
        vmax,vmin,vmed = evec(k,Ixx2,Iyy2,Izz2,Ixy2,Iyz2,Ixz2)
        orbit_x = dotprod(dx_hat[k],dy_hat[k],dz_hat[k],vmin[0],vmin[1],vmin[2])
        orbit_y = dotprod(a_x[k],a_y[k],a_z[k],vmin[0],vmin[1],vmin[2])
        lib_angle2[k] = np.arctan2(orbit_y,orbit_x)
        
        # angle between secondary long body axis and orbit plane
        ltilt =dotprod(no_x[k],no_y[k],no_z[k],vmin[0],vmin[1],vmin[2])
        longtilt2[k] = np.arcsin(np.abs(ltilt))
        
        phi2[k] = np.arctan2(vmin[1],vmin[0])  # secondary long axis angle on xyplane
        # to project vmin onto orbit plane we could use nhat x (vmin x nhat)
        # where (no_x,no_y,no_z) = nhat is the orbit normal
        
        nprec_ang2[k]=np.arctan2(vmax[1],vmax[0])  # secondary short axis projected on xyplane
        
        # angle between secondary short axis and orbit normal
        vtilt =dotprod(no_x[k],no_y[k],no_z[k],vmax[0],vmax[1],vmax[2])
        ntilt2[k] = np.arccos(np.abs(vtilt))
       
        ctheta_NPA  =dotprod(nlx2[k],nly2[k],nlz2[k],vmax[0],vmax[1],vmax[2])
        theta2_NPA[k] = np.arccos(np.abs(ctheta_NPA)) # NPA angle, short axis and angular mom
       
        vmax,vmin,vmed = evec(k,Ixx1,Iyy1,Izz1,Ixy1,Iyz1,Ixz1)
        phi1[k] = np.arctan2(vmin[1],vmin[0])  # primary long axis angle on xyplane
        ctheta_NPA  =dotprod(nlx1[k],nly1[k],nlz1[k],vmax[0],vmax[1],vmax[2])
        theta1_NPA[k]  =np.arccos(np.abs(ctheta_NPA))  #NPA angle,short axis and angular momentum
    
    return tarr,aaarr,eearr,iiarr,lnarr,ararr,maarr,om1,om2,\
        lobliquity_deg1,lobliquity_deg2,sobliquity_deg2,meanmotion,\
        lib_angle2,lprec_ang2,sprec_ang2,nprec_ang2,phi1,phi2,theta_B,theta_k,\
        theta2_NPA,ntilt2,longtilt2
    
    
# at index j from moments of inertia arrays
# return eigenvector of max eigen value
#    and eigenvector of min eigen value
#    and eigenvector of middle eigen value
# should now work if some eigenvalues are same as others
# these are the principal body axes
def evec(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    Imat = np.matrix([[Ixx[j],Ixy[j],Ixz[j]],\
         [Ixy[j],Iyy[j],Iyz[j]],[Ixz[j],Iyz[j],Izz[j]]])
    w, v = LA.eig(Imat)  # eigenvecs v are unit length
    jsort = np.argsort(w) # arguments/indices of a sorted array of eigenvalues,  low to high
    jmax = jsort[2]  # index of maximum eigenvalue
    jmin = jsort[0]  # index of minimum eigenvalue
    jmed = jsort[1]  # index of middle  eigenvalue
    vmax = np.squeeze(np.asarray(v[:,jmax]))   # corresponding eigenvector
    vmin = np.squeeze(np.asarray(v[:,jmin]))   # corresponding eigenvector
    vmed = np.squeeze(np.asarray(v[:,jmed]))   # corresponding eigenvector
    return vmax,vmin,vmed
