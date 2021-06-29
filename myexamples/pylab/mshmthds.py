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



#Mesh Methods

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
    
# tests  
#vi = vol_i(squannit,1)
#print(vi)
#vtot = volume_mesh(squannit)
#print(vtot)

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

def nf_mesh(mesh):
    f = mesh.faces
    print('number of faces ',len(f))