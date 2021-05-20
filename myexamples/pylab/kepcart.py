import numpy as np

######################################
# this file contains
#   ecc_ano(e,l): solve Kepler's equation
#   ecc_anohyp(e,l):solve Kepler's equation hyperbolic case
# orbital elements to cartesian phase space coordinates:
#   cartesian(GM, a, e, i, longnode, argperi, meananom) 
# cartesian phase space to orbital elements
#   keplerian(GM,x,y,z,xd,yd,zd) 
######################################


# solve Kepler's equation, e<1 case
PREC_ecc_ano=1e-16  # precision
def ecc_ano(e,l):
    du=1.0;
    u0 = l + e*np.sin(l) + 0.5*e*e*np.sin(2.0*l); # first guess
    #also see M+D equation 2.55
    # supposed to be good to second order in e, from Brouwer+Clemence              
    counter=0;
    while (np.abs(du) > PREC_ecc_ano):
        l0 = u0 - e*np.sin(u0);  # Kepler's equation here!
        du = (l - l0)/(1.0 - e*np.cos(u0));
        u0 += du;  # this gives a better guess 
        counter = counter + 1
        if (counter > 10000): 
            break;  
        # equation 2.58 from M+D
        #print(du)
    
    return u0;
# kepler's equation is M = E - e sin E
# here l is M and we want to solve for E (eccentric anomali)

#to test:
#u0 = ecc_ano(e,l)
#print(l, u0 - e*np.sin(u0)) for checking accuracy, it works!


# hyperbolic case
def ecc_anohyp(e,l):
    du=1.0;
    u0 = np.log(2.0*l/e + 1.8); # Danby guess
    counter = 0;
    while(np.abs(du) > PREC_ecc_ano):
        fh = e*np.sinh(u0) - u0 - l;  # Kepler's equation hyperbolic here
        dfh = e*np.cosh(u0) - 1.0;
        du = -fh/dfh;
        u0 = u0 + du;
        counter = counter + 1;
        if (counter > 10000): 
            break;
    
    return u0;

# this things solves M = e sinh(E) - E
# to test:
# u0 = ecc_anohyp(e,l)
# to test:  print(l, e*np.sinh(u0) -u0)


# orbital elements to cartesian phase space coordinates
# parabolic case has not been correctly implemented
def cartesian(GM, a, e, i, longnode, argperi, meananom):
    # solve Kepler's equation, to get eccentric anomali
    if (e<1):
        E0 = ecc_ano(e,meananom);  
    else:
        E0 = ecc_anohyp(e,meananom);
        
        
    if (e<1.0):
        cosE = np.cos(E0);
        sinE = np.sin(E0);
    else: 
        cosE = np.cosh(E0);
        sinE = np.sinh(E0);
        
    a = np.abs(a);
    meanmotion = np.sqrt(GM/(a*a*a));
    foo = np.sqrt(np.abs(1.0 - e*e));
    
    # compute unrotated positions and velocities 
    rovera = (1.0 - e*cosE);
    if (e>1.0): 
        rovera = -1.0*rovera;
        
    x = a*(cosE - e);
    y = foo*a*sinE;
    z = 0.0;
    xd = -a*meanmotion * sinE/rovera;
    yd = foo*a*meanmotion * cosE/rovera;
    zd = 0.0;
    if (e>1.0): 
        x = -1.0*x;
        
    # rotate by argument of perihelion in orbit plane
    cosw = np.cos(argperi);
    sinw = np.sin(argperi);
    xp = x * cosw - y * sinw;
    yp = x * sinw + y * cosw;
    zp = z;
    xdp = xd * cosw - yd * sinw;
    ydp = xd * sinw + yd * cosw;
    zdp = zd;
    
    # rotate by inclination about x axis 
    cosi = np.cos(i);
    sini = np.sin(i);
    x = xp;
    y = yp * cosi - zp * sini;
    z = yp * sini + zp * cosi;
    xd = xdp;
    yd = ydp * cosi - zdp * sini;
    zd = ydp * sini + zdp * cosi;

    # rotate by longitude of node about z axis 
    cosnode = np.cos(longnode);
    sinnode = np.sin(longnode);
    state_x = x * cosnode - y * sinnode;
    state_y = x * sinnode + y * cosnode;
    state_z = z;
    state_xd = xd * cosnode - yd * sinnode;
    state_yd = xd * sinnode + yd * cosnode;
    state_zd = zd;
    return state_x, state_y, state_z, state_xd, state_yd, state_zd

  
# cartesian phase space to orbital elements
def keplerian(GM,x,y,z,xd,yd,zd):
    # find direction of angular momentum vector 
    rxv_x = y * zd - z * yd;
    rxv_y = z * xd - x * zd;
    rxv_z = x * yd - y * xd;
    hs = rxv_x*rxv_x + rxv_y*rxv_y + rxv_z*rxv_z;
    h = np.sqrt(hs);
    r = np.sqrt(x*x + y*y + z*z);
    vs = xd*xd + yd*yd + zd*zd;
    rdotv = x*xd + y*yd + z*zd;
    rdot = rdotv/r;

    orbel_i = np.arccos(rxv_z/h);  #inclination!
    if ((rxv_x !=0.0) or (rxv_y !=0.0)): 
        orbel_longnode = np.arctan2(rxv_x, -rxv_y);
    else:
        orbel_longnode = 0.0;

    orbel_a = 1.0/(2.0/r - vs/GM); # semi-major axis could be negative
    
    ecostrueanom = hs/(GM*r) - 1.0;
    esintrueanom = rdot * h/GM;
    # eccentricity
    orbel_e = np.sqrt(ecostrueanom*ecostrueanom + esintrueanom*esintrueanom);
    
    if ((esintrueanom!=0.0) or (ecostrueanom!=0.0)):
        trueanom = np.arctan2(esintrueanom, ecostrueanom);
    else:
        trueanom = 0.0
        
    cosnode = np.cos(orbel_longnode);
    sinnode = np.sin(orbel_longnode);
    
    # u is the argument of latitude 
    if (orbel_i == np.pi/2.0):  # this work around not yet tested
        u = 0.0
    else:
        rcosu = x*cosnode + y*sinnode;
        rsinu = (y*cosnode - x*sinnode)/np.cos(orbel_i);
        # this will give an error if i is pi/2 *******!!!!!!!!
        if ((rsinu!=0.0) or (rcosu!=0.0)): 
            u = np.arctan2(rsinu, rcosu);
        else:
            u = 0.0;

    orbel_argperi = u - trueanom;  # argument of pericenter
    
    # true anomaly to mean anomaly
    foo = np.sqrt(np.abs(1.0 - orbel_e)/(1.0 + orbel_e));
    if (orbel_e <1.0):
        eccanom = 2.0 * np.arctan(foo*np.tan(trueanom/2.0));
        orbel_meananom = eccanom - orbel_e * np.sin(eccanom);
    else:
        eccanom = 2.0 * np.arctanh(foo*np.tan(trueanom/2.0));
        orbel_meananom = orbel_e*np.sinh(eccanom) - eccanom;
  
    # adjust argperi to [-pi,pi]
    if (orbel_argperi > np.pi): 
        orbel_argperi = orbel_argperi - 2.0*np.pi;
    if (orbel_argperi < -np.pi): 
        orbel_argperi = orbel_argperi + 2.0*np.pi;

    return orbel_a,orbel_e,orbel_i,orbel_longnode,orbel_argperi,orbel_meananom


    
