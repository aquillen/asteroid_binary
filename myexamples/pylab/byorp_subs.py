import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

import pymesh
#https://pymesh.readthedocs.io/en/latest/basic.html
# https://github.com/PyMesh/PyMesh

import random
import quaternion
#https://quaternion.readthedocs.io/en/latest/

import meshplot
# for display of meshes
#https://skoch9.github.io/meshplot/tutorial/



# perturb a pymesh sphere (mesh, premade) and stretch it so that
# it becomes an ellipsoid.
#    We can't directly edit vertices or faces
#    see this:  https://github.com/PyMesh/PyMesh/issues/156
#    the work around is to copy the entire mesh after modifying it
# arguments:
#   devrand,  Randomly add devrand to x,y,z positions of each vertex
#     a uniform ditns in [-1,1] is used
#   aratio1 and aratio2,  stretch or compress a sphere by aratio1 and aratio2
# returns: a new mesh
# we assume that longest semi-major axis a is along x,
#    medium semi-axis b is along y, semi-minor c axis is along z
# Volume should stay the same!
# this routine should work on any shaped body
def sphere_perturb(sphere,devrand,aratio1,aratio2):
    #devrand = 0.05  # how far to perturb each vertex
    nv = len(sphere.vertices)
    f = sphere.faces
    v = np.copy(sphere.vertices)
    # add perturbations to x,y,z to all vertices
    for i in range(nv):
        dx = devrand*random.uniform(-1,1)
        dy = devrand*random.uniform(-1,1)
        dz = devrand*random.uniform(-1,1)
        v[i,0] += dx
        v[i,1] += dy
        v[i,2] += dz
        
    
    # aratio1 = c/a  this gives c = aratio1*a
    # aratio2 = b/a  this gives b = aratio2*a
    # volume = 4/3 pi a*b*c for an ellipsoid
    # vol = 1*aratio1*aratio2
    # rad_cor = pow(vol,-1./3.)
    # v[:,2] *= aratio1*rad_cor # make oblate, adjusts z coords
    # v[:,1] *= aratio2*rad_cor # make elongated in xy plane , adjusts y coords
    #  v[:,0] *= rad_cor # adjusts x coords
    # volume should now stay the same
        
    sub_com(v) # subtract center of mass from vertex positions
    psphere = pymesh.form_mesh(v, f)
    psphere.add_attribute("face_area")
    psphere.add_attribute("face_normal")
    psphere.add_attribute("face_centroid")
    
    sbody = body_stretch(psphere,aratio1,aratio2)  # do the stretching
    return sbody
                         

# stretch a mesh body by axis ratios
# arguments:
#    body: mesh
#    aratio1: c/a
#    aratio2: b/a
# returns: a new mesh
# we assume that longest semi-major axis a is along x,
#    medium semi-axis b is along y, semi-minor c axis is along z
# Volume should stay the same!
def body_stretch(body,aratio1,aratio2):
    nv = len(body.vertices)
    f = body.faces
    v = np.copy(body.vertices)
    
    # aratio1 = c/a  this gives c = aratio1*a
    # aratio2 = b/a  this gives b = aratio2*a
    # volume = 4/3 pi a*b*c for an ellipsoid
    vol = 1*aratio1*aratio2
    rad_cor = pow(vol,-1./3.)
    v[:,2] *= aratio1*rad_cor # make oblate, adjusts z coords
    v[:,1] *= aratio2*rad_cor # make elongated in xy plane , adjusts y coords
    v[:,0] *= rad_cor # adjusts x coords
    # volume should now stay the same
    
    sub_com(v) # subtract center of mass from vertex positions
    sbody = pymesh.form_mesh(v, f)
    sbody.add_attribute("face_area")
    sbody.add_attribute("face_normal")
    sbody.add_attribute("face_centroid")
    return sbody
    

# substract the center of mass from a list of vertices
def sub_com(v):
    nv = len(v)
    xsum = np.sum(v[:,0])
    ysum = np.sum(v[:,1])
    zsum = np.sum(v[:,2])
    xmean = xsum/nv
    ymean = ysum/nv
    zmean = zsum/nv
    v[:,0]-= xmean
    v[:,1]-= ymean
    v[:,2]-= zmean
    
# compute surface area by summing area of all facets
# divide by 4pi which is the surface area of a sphere with radius 1
def surface_area(mesh):
    #f = mesh.faces
    S_i = mesh.get_face_attribute('face_area')
    area =np.sum(S_i)
    return area/(4*np.pi)


# print number of faces
def nf_mesh(mesh):
    f = mesh.faces
    print('number of faces ',len(f))


# show a mesh using meshplot with a bounding square in the image
def plt_mesh_square(vertices,faces,xmax):
    #m = np.array([-xmax,-xmax,-xmax])
    #ma = np.abs(m)

    # Corners of the bounding box
    v_box = np.array([[-xmax, -xmax, 0], [-xmax, xmax, 0], [xmax, xmax,0] , [xmax, -xmax, 0]])

    # Edges of the bounding box
    f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int)

    p = meshplot.plot(vertices, faces, return_plot=True)  # plot body

    p.add_edges(v_box, f_box, shading={"line_color": "red"});
    #p.add_points(v_box, shading={"point_color": "green"})
    return p



# perform a rotation on a vertex list and return a new set of rotated vertices
# rotate about axis and via angle in radians
def rotate_vertices(vertices,axis,angle):
#     v = np.copy(vertices)
#     ### if using pymesh.Quaternion
#     qs = pymesh.Quaternion.fromAxisAngle(axis, angle)
#     nv = len(v)
#     ### loop over all vertices and do the rotation
#     for i in range(nv):
#          v[i] = qs.rotate(v[i]) ### perform rotation
#     return v
    ## if using  np.quaternion
    v = np.copy(vertices)
    qs = quaternion.from_rotation_vector(rot=axis*angle)
    vp=quaternion.rotate_vectors(qs, v) #####, axis=-1)  # rotate vectors
    return vp


# compute cross product C=AxB using components
def cross_prod_xyz(Ax,Ay,Az,Bx,By,Bz):
    Cx = Ay*Bz - Az*By
    Cy = Az*Bx - Ax*Bz
    Cz = Ax*By - Ay*Bx
    return Cx,Cy,Cz

# compute face_areas and face normals from a face list and a vertex list
def face_areas(vertices,faces):
    nf = len(faces)
    S_i = np.zeros(nf)
    f_normals = np.zeros((nf,3))
    for iface in range(nf):
        iv1 = faces[iface,0]  # indexes of the 3 vertices
        iv2 = faces[iface,1]
        iv3 = faces[iface,2]
        v1 = vertices[iv1]  # the 3 vertices
        v2 = vertices[iv2]
        v3 = vertices[iv3]
        e1 = v2 - v1   # edge vectors
        e2 = v3 - v2
        ax,ay,az=cross_prod_xyz(e1[0],e1[1],e1[2],e2[0],e2[1],e2[2])  # cross product
        area = np.sqrt(ax**2 + ay**2 + az**2)  # area of parallelogram
        S_i[iface] = 0.5*area #  1/2 area of parallelogram spanned by e1, e2 is area of triangle
        f_normals[iface,0] = ax/area  #normalize vector cross product
        f_normals[iface,1] = ay/area
        f_normals[iface,2] = az/area
        # signs checked by plotting normal components in color
    return S_i,f_normals

# compute the volume of the tetrahedron formed from face with index iface
# and the origin
def vol_i(mesh,iface):
    f = mesh.faces
    v = mesh.vertices
    iv1 = f[iface,0]  # indexes of the 3 vertices
    iv2 = f[iface,1]
    iv3 = f[iface,2]
    #print(iv1,iv2,iv3)
    v1 = v[iv1]  # the 3 vertices
    v2 = v[iv2]
    v3 = v[iv3]
    #print(v1,v2,v3)
    mat = np.array([v1,v2,v3])
    # the volume is equal to 1/6 determinant of the matrix formed with the three vertices
    # https://en.wikipedia.org/wiki/Tetrahedron
    #print(mat)
    vol = np.linalg.det(mat)/6.0  # compute determinant
    return vol

# compute the volume of the mesh by looping over all tetrahedrons formed from the faces
# we assume that the body is convex
def volume_mesh(mesh):
    f = mesh.faces
    nf = len(f)
    vol = 0.0
    for iface in range(nf):
        vol += vol_i(mesh,iface)
    return vol
        
# if vol equ radius is 1  the volume should be equal to 4*np.pi/3 which is 4.1888
    

# correct all the radii so that the volume becomes that of a sphere with radius 1
# return a new mesh
def cor_volume(mesh):
    vol = volume_mesh(mesh)
    print('Volume {:.4f}'.format(vol))
    rad = pow(vol*3/(4*np.pi),1.0/3.0)
    print('radius of vol equ sphere {:.4f}'.format(rad))
    f = mesh.faces
    v = np.copy(mesh.vertices)
    v /= rad
    newmesh = pymesh.form_mesh(v, f)
    newmesh.add_attribute("face_area")
    newmesh.add_attribute("face_normal")
    newmesh.add_attribute("face_centroid")
    vol = volume_mesh(newmesh)
    print('new Volume {:.3f}'.format(vol))
    return newmesh
    
# compute the radiation force instantaneously on a triangular mesh for each facit
# arguments:
#     S_i  vector of face areas
#     f_normal vector of face normals
#     s_hat is a 3 length np.array (a unit vector) pointing to the Sun
# return the vector F_i for each facet
# returns:  F_i_x is the x component of F_i and is a vector that has the length of the number of faces
# Force is zero if facets are not on the day side
def F_i(S_i,f_normal,s_hat):
    s_len = np.sqrt(s_hat[0]**2 + s_hat[1]**2 + s_hat[2]**2)  # in case s_hat was not normalized
    
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
