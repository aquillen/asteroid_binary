import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pymesh
#https://pymesh.readthedocs.io/en/latest/basic.html
import time
import multiprocessing
import numexpr as ne
import meshplot
# for display of meshes
#https://skoch9.github.io/meshplot/tutorial/

from mshmthds import *  

#YORP/BYORP Helper Methods
# compute the radiation force instantaneously on a triangular mesh for each facit
# arguments:  
#     mesh, the body (a triangular surface mesh)
#     s_hat is a 3 length np.array (a unit vector) pointing to the Sun
# return the vector F_i for each facet
# returns:  F_i_x is the x component of F_i and is a vector that has the length of the number of faces
# Force is zero if facets are not on the day side
def F_i(mesh,s_hat):
    s_len = np.sqrt(s_hat[0]**2 + s_hat[1]**2 + s_hat[2]**2)  # in case s_hat was not normalized
    #nf = len(mesh.faces)
    S_i = mesh.get_face_attribute('face_area')  # vector of facet areas
    f_normal = mesh.get_face_attribute('face_normal')  # vector of vector of facet normals
    # normal components 
    nx = np.squeeze(f_normal[:,0])    # a vector, of length number of facets
    ny = np.squeeze(f_normal[:,1])
    nz = np.squeeze(f_normal[:,2])
    # dot product of n_i and s_hat
    n_dot_s = (nx*s_hat[0] + ny*s_hat[1] + nz*s_hat[2])/s_len  # a vector
    F_i_x = -S_i*n_dot_s*nx #  a vector, length number of facets
    F_i_y = -S_i*n_dot_s*ny
    F_i_z = -S_i*n_dot_s*nz
    ii = (n_dot_s <0)  # the night sides 
    F_i_x[ii] = 0  # get rid of night sides
    F_i_y[ii] = 0
    F_i_z[ii] = 0
    return F_i_x,F_i_y,F_i_z   # these are each vectors for each face

# compute radiation forces F_i for each face, but averaging over all positions of the Sun
# a circular orbit for the asteroid is assumed
# arguments: 
#    nphi_Sun is the number of solar angles, evenly spaced in 2pi so we are assuming circular orbit
#    incl is solar orbit inclination in radians
# returns: F_i_x average and other 2 components of forces for each facet
def F_i_sun_ave(mesh,nphi_Sun,incl):
    dphi = 2*np.pi/nphi_Sun
    # compute the first set of forces so we have vectors the right length
    phi = 0.0
    s_hat = np.array([np.cos(phi)*np.cos(incl),np.sin(phi)*np.cos(incl),np.sin(incl)])
    # compute the radiation force instantaneously on the triangular mesh for sun at s_hat
    F_i_x_sum,F_i_y_sum,F_i_z_sum = F_i(mesh,s_hat)
    # now compute the forces for the rest of the solar angles
    for i in range(1,nphi_Sun): # do the rest of the angles
        phi = i*dphi
        s_hat = np.array([np.cos(phi)*np.cos(incl),np.sin(phi)*np.cos(incl),np.sin(incl)])
        # compute the radiation force instantaneously on the triangular mesh for sun at s_hat
        F_i_x,F_i_y,F_i_z = F_i(mesh,s_hat)  # These are vectors of length number of facets
        F_i_x_sum += F_i_x  # sum up forces
        F_i_y_sum += F_i_y
        F_i_z_sum += F_i_z
    F_i_x_ave = F_i_x_sum/nphi_Sun  # average
    F_i_y_ave = F_i_y_sum/nphi_Sun
    F_i_z_ave = F_i_z_sum/nphi_Sun
    return F_i_x_ave,F_i_y_ave,F_i_z_ave  # these are vectors for each face

# compute cross product C=AxB using components
def cross_prod_xyz(Ax,Ay,Az,Bx,By,Bz):
    Cx = Ay*Bz - Az*By
    Cy = Az*Bx - Ax*Bz
    Cz = Ax*By - Ay*Bx
    return Cx,Cy,Cz

# compute total Yorp torque averaging over nphi_Sun solar positions
# this is at a single body orientation
# a circular orbit is assumed
# arguments:
#   mesh: the body
#   nphi_Sun is the number of solar angles
#   incl is solar orbit inclination in radians
# returns: torque components
def tau_Ys(mesh,nphi_Sun,incl):
    # compute F_i for each face, but averaging over all positions of the Sun
    F_i_x_ave, F_i_y_ave,F_i_z_ave = F_i_sun_ave(mesh,nphi_Sun,incl)
    r_i = mesh.get_face_attribute("face_centroid")  # radii to each facet
    rx = np.squeeze(r_i[:,0])  # radius of centroid from center of mass
    ry = np.squeeze(r_i[:,1])  # these are vectors, length number of faces
    rz = np.squeeze(r_i[:,2])
    # cross product works on vectors
    tau_i_x,tau_i_y,tau_i_z = cross_prod_xyz(rx,ry,rz,F_i_x_ave,F_i_y_ave,F_i_z_ave)
    #This is the torque from each day lit facet
    tau_x = np.sum(tau_i_x)  # sum up forces from all faces
    tau_y = np.sum(tau_i_y)
    tau_z = np.sum(tau_i_z)
    return tau_x,tau_y,tau_z  # these are numbers for torque components

# compute total BYORP averaging over nphi_Sun solar positions
# for a single binary vector a_bin and body position described with mesh
# arguments:
#    incl is solar orbit inclination in radians
#    nphi_Sun is the number of solar angles
# returns: torque components
def tau_Bs(mesh,nphi_Sun,incl,a_bin):
    # compute F_i for each face, but averaging over all positions of the Sun
    F_i_x_ave, F_i_y_ave,F_i_z_ave = F_i_sun_ave(mesh,nphi_Sun,incl)  # these are vectors length number of faces
    # forces from day lit faces
    F_x = np.sum(F_i_x_ave)  #sum up the force
    F_y = np.sum(F_i_y_ave)
    F_z = np.sum(F_i_z_ave)
    a_x = a_bin[0]  # binary direction
    a_y = a_bin[1]
    a_z = a_bin[2]
    tau_x,tau_y,tau_z = cross_prod_xyz(a_x,a_y,a_z,F_x,F_y,F_z) # cross product
    return tau_x,tau_y,tau_z  # these are numbers that give the torque components
        
    
    
# first rotate vertices in the mesh about the z axis by angle phi in radians
# then tilt over the body by obliquity which is an angle in radians
# arguments:
#    mesh, triangular surface mess for body
#    obliquity, angle in radius to tilt body z axis over
#    phi, angle in radians to spin/rotate body about its z axis
#    phi_prec,  angle in randias that tilt is done, it's a precession angle
#      sets rotation axis for tilt, this axis is in the xy plane
# returns: 
#     new_mesh: the tilted/rotated mesh
#     zrot:  the new z-body spin axis
def tilt_obliq(mesh,obliquity,phi,phi_prec):
    f = mesh.faces
    v = np.copy(mesh.vertices)
    nv = len(v)
    
    # precession angle is phi_prec
    axist = np.array([np.cos(phi_prec),np.sin(phi_prec),0])  
    qt = pymesh.Quaternion.fromAxisAngle(axist, obliquity)
    zaxis = np.array([0,0,1])
    
    zrot = qt.rotate(zaxis) # body principal axis will become zrot
    
    # spin rotation about now tilted principal body axis 
    qs = pymesh.Quaternion.fromAxisAngle(zrot, phi)
    
    # loop over all vertices and do two rotations
    for i in range(nv):
        v[i] = qt.rotate(v[i]) # tilt it over
        v[i] = qs.rotate(v[i]) # spin
    
    new_mesh = pymesh.form_mesh(v, f)
    new_mesh.add_attribute("face_area")
    new_mesh.add_attribute("face_normal")
    new_mesh.add_attribute("face_centroid")
    
    return new_mesh,zrot
    

# tilt,spin a body and compute binary direction, assuming tidally locked
# arguments:
#   body:  triangular surface mesh (in principal axis coordinate system)
#   nphi is the number of angles that could be done with indexing by iphi
#   obliquity:  w.r.t to binary orbit angular momentum direction
#   iphi:  number of rotations by dphi where dphi = 2pi/nphi
#      this is principal axis rotation about z axis
#   phi0: an offset for phi applied to body but not binary axis
#   phi_prec: a precession angle for tilting
# returns: 
#   tbody, a body rotated  after iphi rotations by dphi and tilted by obliquity
#   a_bin, binary direction assuming same rotation rate, tidal lock
#   l_bin:  binary orbit angular momentum orbital axis
#   zrot:  spin axis direction 
def tilt_and_bin(body,obliquity,nphi,iphi,phi0,phi_prec):
    dphi = 2*np.pi/nphi
    phi = iphi*dphi 
    tbody,zrot = tilt_obliq(body,obliquity,phi + phi0,phi_prec)  # tilt and spin body
    a_bin = np.array([np.cos(phi),np.sin(phi),0.0])   # direction to binary
    l_bin = np.array([0,0,1.0])  # angular momentum axis of binary orbit
    return tbody,a_bin,l_bin,zrot


# compute the YORP torque on body
# arguments:
#   body:  triangular surface mesh (in principal axis coordinate system)
#   nphi is number of body angles spin
#   nphi_Sun is the number of solar angles used
#   obliquity: angle of body w.r.t to Sun aka ecliptic pole
# returns: 
#   3 torque components 
#   torque dot spin axis so spin down rate can be computed
#   torque dot azimuthal unit vector so obliquity change rate can be computed
def compute_Y(body,obliquity,nphi,nphi_Sun):
    incl = 0.0  # set Sun inclination to zero so obliquity is w.r.t solar orbit
    phi0 = 0  # offset in spin set to zero
    phi_prec=0  # precession angle set to zero
    tau_Y_x = 0.0
    tau_Y_y = 0.0
    tau_Y_z = 0.0
    for iphi in range(nphi):  # body positions
        # rotate the body and tilt it over
        tbody,a_bin,l_bin,zrot = tilt_and_bin(body,obliquity,nphi,iphi,phi0,phi_prec)
        # compute torques over solar positions
        tau_x,tau_y,tau_z = tau_Ys(tbody,nphi_Sun,incl)
        tau_Y_x += tau_x
        tau_Y_y += tau_y
        tau_Y_z += tau_z
        
    tau_Y_x /= nphi  # average
    tau_Y_y /= nphi
    tau_Y_z /= nphi
    # compute component that affects spin-down/up rate, this is tau dot spin axis
    sx = zrot[0]; sy = zrot[1]; sz=zrot[2]
    tau_s = tau_Y_x*sx + tau_Y_y*sy + tau_Y_z*sz
    # we need a unit vector, phi_hat, that is in the xy plane, points in the azimuthal direction
    # and is perpendicular to the rotation axis
    spl = np.sqrt(sx**2 + sy**2)
    tau_o = 0
    if (spl >0):
        phi_hat_x =  sy/spl
        phi_hat_y = -sx/spl
        phi_hat_z = 0
        tau_o = tau_Y_x*phi_hat_x + tau_Y_y*phi_hat_y+tau_Y_z*phi_hat_z
        # tau_o should tell us about obliquity change rate
    return tau_Y_x,tau_Y_y,tau_Y_z,tau_s,tau_o 

# compute the BYORP torque, for a tidally locked binary
# arguments:
#   body:  triangular surface mesh (in principal axis coordinate system)
#   nphi is the number of body angles we will use (spin)
#   obliquity is body tilt w.r.t to binary orbit
#   incl is solar orbit inclination 
#   nphi_Sun is the number of solar angles used
#   phi0 an offset for body spin angle that is not applied to binary direction
#   phi_prec  z-axis precession angle, relevant for obliquity 
# returns:
#   3 torque components
#   torque dot l_bin so can compute binary orbit drift rate
def compute_BY(body,obliquity,nphi,nphi_Sun,incl,phi0,phi_prec):
    tau_BY_x = 0.0
    tau_BY_y = 0.0
    tau_BY_z = 0.0
    for iphi in range(nphi):  # body positions
        # rotate the body and tilt it over, and find binary direction
        tbody,a_bin,l_bin,zrot = tilt_and_bin(body,obliquity,nphi,iphi,phi0,phi_prec)
        # a_bin is binary direction
        # compute torques over spin/body positions
        tau_x,tau_y,tau_z =tau_Bs(tbody,nphi_Sun,incl,a_bin)
        tau_BY_x += tau_x
        tau_BY_y += tau_y
        tau_BY_z += tau_z
        
    tau_BY_x /= nphi  # average
    tau_BY_y /= nphi
    tau_BY_z /= nphi
    # compute component that affects binary orbit angular momentum
    # this is tau dot l_bin
    tau_l = tau_BY_x*l_bin[0] + tau_BY_y*l_bin[1] + tau_BY_z*l_bin[2] 
    return tau_BY_x,tau_BY_y,tau_BY_z, tau_l 