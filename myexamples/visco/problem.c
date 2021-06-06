

/**
 * resolved mass spring model
 * using the leap frog integrator. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "rebound.h"
#include "tools.h"
#include "output.h"
#include "spring.h"


int NS;  // global numbers of springs
struct spring* springs;
void reb_springs();// to pass springs to display

double gamma_all; // for gamma  of all springs
double t_damp;    // end faster damping, relaxation
double t_print;   // for table printout 
char froot[30];   // output files
int npert=1; // number of perturbative point masses

int icentral=0; // central mass location

#define NPMAX 10  // maximum number of point masses
double itaua[NPMAX],itaue[NPMAX]; // inverse of migration timescales
double itmig[NPMAX];  // inverse timescale to get rid of migration

void heartbeat(struct reb_simulation* const r);

// quadrupole of planet 
double J2_plus = 0.0; // is greater than zero for an oblate planet
double R_plus = 1.0;
   

void additional_forces(struct reb_simulation* r){
   spring_forces(r); // spring forces
   // double C20 = -J2_plus;
   // quadforce(r, C20, 0.0, 0.0, R_plus, 0.0,r->N-npert);
}


int main(int argc, char* argv[]){
	struct reb_simulation* const r = reb_create_simulation();
        struct spring spring_mush; // spring parameters for mush
	// Setup constants
	r->integrator	= REB_INTEGRATOR_LEAPFROG;
	r->gravity	= REB_GRAVITY_BASIC;
	r->boundary	= REB_BOUNDARY_NONE;
	r->G 		= 1;		
        r->additional_forces = additional_forces;  // setup callback function for additional forces
        double mball = 1.0;          // total mass of ball
        double rball = 1.0;          // radius of a ball
        double tmax = 0.0;  // if 0 integrate forever

// things to set!  can be read in with parameter file
        double dt; 
        double b_distance,omegaz,ks,mush_fac,gamma_fac;
        double ratio1,ratio2,obliquity_deg;
        int lattice_type;
        double rad[NPMAX],mp[NPMAX];
        double aa[NPMAX],ee[NPMAX],ii[NPMAX];
        double longnode[NPMAX],argperi[NPMAX],meananom[NPMAX];
        int npointmass=0;
        unsigned int seed=1;

    if (argc ==1){
        strcpy(froot,"t1");   // to make output files
	dt	   = 1e-3;    // Timestep
        tmax       = 0.0;     // max integration time
        t_print    = 1.0;     // printouts for table
	lattice_type  = 0;    // 0=rand 1=hcp
        b_distance = 0.15;    // min separation between particles
        mush_fac    = 2.3;    // ratio of smallest spring distance to minimum interparticle dist
        ks          = 8e-2;   // spring constant
        // spring damping
        gamma_all   = 1.0;    // final damping coeff
        gamma_fac   = 5.0;    // factor initial gamma is higher that gamma_all
        t_damp      = 1.0;    // gamma from initial gamma 
                              // to gamma_all for all springs at this time
        ratio1      = 1.0;    // axis ratio resolved body   b/a
        ratio2      = 1.0;    // axis ratio c/a
        omegaz      = 0.2;    // initial spin
        obliquity_deg = 0.0;  // obliquity

        npointmass=1;  // add one point mass
        int ip=0;   // index
        mp[ip]       = 1.0;   // mass
        rad[ip]      = 0.5;   // display radius
        itaua[ip]    = 0.0;   // inverse drift rate in a
        itaue[ip]    = 0.0;   // inverse drift rate in e
        itmig[ip]    = 0.0;   // get rid of drift rate in inverse of this time
	// orbit
        aa[ip]       =10.0;    // distance of m1 from resolved body, semi-major orbital
	ee[ip]       =0.0;    // initial eccentricity
	ii[ip]       =0.0;    // initial inclination 
	argperi[ip]  =0.0;    // initial orbtal elements
	longnode[ip] =0.0;    // initial 
	meananom[ip] =0.0;    // initial 

     }
     else{
        FILE *fpi;
        fpi = fopen(argv[1],"r");
        char line[300];
        fgets(line,300,fpi);  sscanf(line,"%s",froot);   // fileroot for outputs
        fgets(line,300,fpi);  sscanf(line,"%lf",&dt);    // timestep
        fgets(line,300,fpi);  sscanf(line,"%lf",&tmax);  // integrate to this time
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_print); // output timestep
        fgets(line,300,fpi);  sscanf(line,"%d",&lattice_type); // particle lattice type
        fgets(line,300,fpi);  sscanf(line,"%lf",&b_distance); // min interparticle distance
        fgets(line,300,fpi);  sscanf(line,"%lf",&mush_fac);   // sets max spring length
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks);         // spring constant
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_fac);  // factor initial gamma is higher
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_all);  // damping final
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_damp);     // time to switch
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio1);     // axis ratio for body b/a
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio2);     // second axis ratio   c/a
        fgets(line,300,fpi);  sscanf(line,"%lf",&omegaz);     // initial body spin
        fgets(line,300,fpi);  sscanf(line,"%lf",&obliquity_deg); // obliquity degrees
        fgets(line,300,fpi);  sscanf(line,"%u",&seed); // random number seed, if 1 then use time of day      
        fgets(line,300,fpi);  sscanf(line,"%lf %lf",&J2_plus,&R_plus); // for an oblate planet
        fgets(line,300,fpi);  sscanf(line,"%d",&npointmass); // number of point masses
        for (int ip=0;ip<npointmass;ip++){
           fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf %lf %lf",
             mp+ip,rad+ip,itaua+ip,itaue+ip,itmig+ip); 
           fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf %lf %lf %lf",
             aa+ip,ee+ip,ii+ip,longnode+ip,argperi+ip,meananom+ip);
        }

     }
     double obliquity = obliquity_deg*M_PI/180.0;   // in radians
     npert = 0;

/// end parameters of things to set /////////////////////////

        r->dt=dt; // set integration timestep
	const double boxsize = 3.2*rball;    // display
	reb_configure_box(r,boxsize,1,1,1);
	r->softening      = b_distance/100.0;	// Gravitational softening length


   // properties of springs
   spring_mush.gamma     = gamma_fac*gamma_all; // initial damping coefficient
   spring_mush.ks        = ks; // spring constant
   // spring_mush.smax      = 1e6; // not used currently
   double mush_distance=b_distance*mush_fac; 
       // distance for connecting and reconnecting springs

   
   FILE *fpr;
   char fname[200];
   sprintf(fname,"%s_run.txt",froot);
   fpr = fopen(fname,"w");

   NS=0; // start with no springs 

// for volume to be the same, adjusting here!!!!
   double volume_ratio = pow(rball,3.0)*ratio1*ratio2;  // neglecting 4pi/3 factor
   double vol_radius = pow(volume_ratio,1.0/3.0);

   rball /= vol_radius; // volume radius used to compute body semi-major axis
           // assuming that semi-major axis is rball
   // fprintf(fpr,"vol_ratio %.6f\n",volume_ratio); // with respect to 4pi/3 
   fprintf(fpr,"a %.3f\n",rball); 
   fprintf(fpr,"b %.3f\n",rball*ratio1); 
   fprintf(fpr,"c %.3f\n",rball*ratio2); 
   volume_ratio = pow(rball,3.0)*ratio1*ratio2;  // neglecting 4pi/3 factor
   // fprintf(fpr,"vol_ratio %.6f\n",volume_ratio); // with respect to 4pi/3 
   // so I can check that it is set to 1

   if (seed != 1 )  // seed random number generator
      srand(seed);
   // create resolved body particle distribution
   if (lattice_type==0){
      // rand_football_from_sphere(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
      rand_football(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
   }
   if (lattice_type ==1){
      fill_hcp(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }
   if (lattice_type ==2){
      fill_cubic(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }


   int il=0;  // index range for resolved body
   int ih=r->N;

   subtractcom(r,il,ih);  // move reference frame to resolved body 

   rotate_to_principal(r, il, ih); // rotate to principal axes

   // spin it
   subtractcov(r,il,ih); // center of velocity subtracted 
   spin(r,il, ih, 0.0, 0.0, omegaz);  // change one of these zeros to tilt it!
       // can spin about non principal axis
   subtractcov(r,il,ih); // center of velocity subtracted 
   double speriod  = fabs(2.0*M_PI/omegaz);
   printf("spin period %.6f\n",speriod);
   fprintf(fpr,"spin period %.6f\n",speriod);
   rotate_body(r, il, ih, 0.0, obliquity, 0.0); // tilt by obliquity

   // make springs, all pairs connected within interparticle distance mush_distance
   connect_springs_dist(r,mush_distance, il, ih, spring_mush);

   // assume minor semi is rball*ratio2
   double ddr = rball*ratio2 - 0.5*mush_distance;
   ddr = 0.4; // mini radius  for computing Young modulus
   double Emush = Young_mush(r,il,ih, 0.0, ddr); // compute from springs out to ddr
   double Emush_big = Young_mush_big(r,il,ih);
   printf("ddr = %.3f mush_distance =%.3f\n",ddr,mush_distance);
   printf("Youngs_modulus %.6f\n",Emush);
   printf("Youngs_modulus_big %.6f\n",Emush_big);
   fprintf(fpr,"Youngs_modulus %.6f\n",Emush);
   fprintf(fpr,"Youngs_modulus_big %.6f\n",Emush_big);
   fprintf(fpr,"mush_distance %.4f\n",mush_distance);
   double LL = mean_L(r);  // mean spring length
   printf("mean_L  %.4f\n",LL);
   fprintf(fpr,"mean_L %.4f\n",LL);
   // relaxation timescale
   // note no 2.5 here!
   double tau_relax = 1.0*gamma_all*0.5*(mball/(r->N -npert))/spring_mush.ks; // Kelvin Voigt relaxation time
   printf("relaxation_time %.3e\n",tau_relax);
   fprintf(fpr,"relaxation_time %.3e\n",tau_relax);

   double om = 0.0; 
   if (npointmass >0){ 
      // set up central star
      int ip=0;
       om = add_pt_mass_kep(r, il, ih, -1, mp[ip], rad[ip],
           aa[ip],ee[ip], ii[ip], longnode[ip],argperi[ip],meananom[ip]);
      fprintf(fpr,"resbody_mm %.3f\n",om);
      fprintf(fpr,"resbody_period %.2f\n",2.0*M_PI/om);
      printf("resbody_mm=%.3f resbody_period=%.2f\n",om,2.0*M_PI/om);
      icentral = ih;
      // set up rest of point masses
      for(int ipp = 1;ipp<npointmass;ipp++){
          double omp; // central mass assumed to be first one ipp=ih = icentral
          omp = add_pt_mass_kep(r, il, ih, icentral, mp[ipp], rad[ipp],
             aa[ipp],ee[ipp], ii[ipp], longnode[ipp],argperi[ipp],meananom[ipp]);
          printf("pointm_%d_mm %.4f\n",ipp,omp);
          printf("pointm_%d_period %.2f\n",ipp,2.0*M_PI/omp);
          fprintf(fpr,"pointm_%d_mm %.4f\n",ipp,omp);
          fprintf(fpr,"pointm_%d_period %.2f\n",ipp,2.0*M_PI/omp);
      }
      npert = npointmass;
   }
   // printf("varpi_precession_fac=%.3e\n", 1.5*J2_plus*pow(R_plus/aa[0],2.0));

  // this is probably not correct if obliquity is greater than pi/2
   double barchi = 2.0*fabs(om - omegaz)*tau_relax;  // initial value of barchi
   double posc = 0.5*2.0*M_PI/fabs(om - omegaz); // for oscillations!
   fprintf(fpr,"barchi %.4f\n",barchi);
   printf("barchi %.4f\n",barchi);
   fprintf(fpr,"posc %.6f\n",posc);
   if (npert >0){
      double R_H  = aa[0]*pow(3*mp[0],-1.0/3.0);  // compute Hill radius
      printf("R_H= %.3f\n",R_H);
   }

// double na = om*aa;
// double adot = 3.0*m1*na/pow(aa,5.0); // should approximately be adot
// fprintf(fpr,"adot %.3e\n",adot);
   

   // ratio of numbers of particles to numbers of springs for resolved body
   double Nratio = (double)NS/(ih-il);
   printf("N=%d NS=%d NS/N=%.1f\n", r->N, NS, Nratio);
   fprintf(fpr,"N %d\n", r->N);
   fprintf(fpr,"NS %d\n",  NS);
   fprintf(fpr,"NS/N %.1f\n", Nratio);

   if (J2_plus != 0.0){
     fprintf(fpr,"J2_plus %.3e\n", J2_plus);
     fprintf(fpr,"R_plus %.2f\n", R_plus);
   }

   fclose(fpr);

   reb_springs(r); // pass spring index list to display
   r->heartbeat = heartbeat;
   centerbody(r,il,ih);  // move reference frame to resolved body 

   // max integration time
   if (tmax ==0.0)
      reb_integrate(r, INFINITY);
   else
      reb_integrate(r, tmax);
}


#define NSPACE 50
void heartbeat(struct reb_simulation* const r){
        static int first=0;
        static char extendedfile[50];
        static char pointmassfile[NPMAX*NSPACE];
        if (first==0){
           first=1;
           sprintf(extendedfile,"%s_ext.txt",froot);
           for(int i=0;i<1;i++){
             sprintf(pointmassfile+i*NSPACE,"%s_pm%d.txt",froot,i);
           }
        }
	if (reb_output_check(r,10.0*r->dt)){
		reb_output_timing(r,0);
	}
        if (fabs(r->t - t_damp) < 0.9*r->dt) set_gamma(gamma_all,0,r->N-npert); 
            // damp initial bounce only 
            // reset gamma only at t near t_damp
	
         // stuff to do every timestep
         centerbody(r,0,r->N-npert);  // move reference frame to resolved body for display
         
         // int il = 0; // resolved body index range
         // int ih = r->N -npert;

	 if (reb_output_check(r,t_print)) {
            print_extended(r,0,r->N-npert,extendedfile,0.0); // orbital info and stuff
            if (npert>0) 
               for(int i=0;i<1;i++){
                  int ip = icentral+i;
                  print_pm(r,ip,i,pointmassfile+i*NSPACE);
               }
         }

        // if (r->t >t_damp){ }


}

// make a spring index list for display
void reb_springs(struct reb_simulation* const r){
   r->NS = NS;
   r->springs_ii = malloc(NS*sizeof(int));
   r->springs_jj = malloc(NS*sizeof(int));
   for(int i=0;i<NS;i++){
     r->springs_ii[i] = springs[i].i;
     r->springs_jj[i] = springs[i].j;
   }
}

// if number of springs NS increases due to added springs
// we need to update the display for springs
void update_reb_springs(struct reb_simulation* const r){
   int NS_old = r->NS;
   r->NS = NS;
   if (NS>NS_old){
      r->springs_ii = realloc(r->springs_ii,NS*sizeof(int));
      r->springs_jj = realloc(r->springs_jj,NS*sizeof(int));
      for(int i=NS_old;i<NS;i++){
        r->springs_ii[i] = springs[i].i;
        r->springs_jj[i] = springs[i].j;
      }
   }

}


// add a free particle to particle list before the npert point masses
// you need to generate the particle position and velocity mass and radius first
void add_free_particle_ptnew(struct reb_simulation* const r, struct reb_particle ptnew ){
   int N = r->N; 
   struct reb_particle pt;
   pt.x = 0; pt.y = 0; pt.z=0; pt.vx = 0; pt.vy = 0; pt.vz=0; pt.ax = 0; pt.ay = 0; pt.az=0;
   pt.r = 0.1; pt.m = 0; 
   reb_add(r,pt); // place holder number of particles  increases
   int ip=1;
   for (ip=1;ip<=npert;ip++){ // move all the point mass particles up one
      pt = r->particles[N-ip];
      r->particles[N-ip+1] = pt; 
      // printf("ip=%d N=%d npert=%d N-ip=%d N-ip+1=%d\n",ip,N,npert,N-ip,N-ip+1);
   }
   r->particles[N-npert] = ptnew;  // add the new particle
   // printf("ptnew ip=%d N=%d N-ip=%d\n",ip,N,N-ip);
   npert++;  // it's not part of the resolved body, must be global!
}

// generate a new particle near the satellite
// ds is distance in orbit in frame of satellite from satellite
// dr is radius from central planet 
// initial condition is corotating circular orbit
struct reb_particle gen_ptnew(struct reb_simulation* const r, double ds, double dr){
   struct reb_particle pt;
   pt.ax = 0; pt.ay = 0; pt.az = 0;
   pt.vz = 0; pt.z = 0;
   struct reb_particle pt0 = r->particles[0]; // a particle in pan
   pt.m = pt0.m; // copy mass and radius
   pt.r = pt0.r;

   int il = 0; int ih=r->N - npert; // resolved body
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc);  // center of mass of resolved body
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body

   struct reb_particle ptc = r->particles[r->N-1]; // the central planet
// assuming resolved body is at origin
   
   double phi = atan2(yc-ptc.y, xc-ptc.x);
   double rc = dr+sqrt(pow(xc-ptc.x,2.0)+ pow(yc-ptc.y,2.0));
   double vc = sqrt(r->G*(ptc.m+1)/rc);
   double dphi = ds/rc;
   double x0  = rc*cos(phi+dphi); // circular orbits
   double y0  = rc*sin(phi+dphi);
   double vx0  =-vc*sin(phi+dphi);
   double vy0  = vc*cos(phi+dphi);
   pt.x = x0 + ptc.x;
   pt.y = y0 + ptc.y;
   pt.vx = vx0 + ptc.vx;
   pt.vy = vy0 + ptc.vy;
//  printf("x0=%0.3f y0=%0.3f vx0=%0.3f vy0=%0.3f rc=%0.3f vc=%0.3f \n", x0,y0,vx0,vy0,rc,vc);
// printf("x=%0.3f y=%0.3f vx=%0.3f vy=%0.3f rc=%0.3f vc=%0.3f \n", pt.x,pt.y,pt.vx,pt.vy,rc,vc);
   return pt;
} 

// generate a new free particle and add it to simulation
void add_free_particle(struct reb_simulation* const r, double ds,double dr){
    struct reb_particle ptnew = gen_ptnew(r, ds,dr); // generate free particle in orbit
    printf("x,y =%.3f %.3f\n",ptnew.x,ptnew.y);  
    add_free_particle_ptnew(r, ptnew );  // add it to simulation
}


// return distance between particles i,j
double dist_ij(struct reb_simulation* const r, int i, int j){
    struct reb_particle pti  = r->particles[i];
    struct reb_particle ptj  = r->particles[j];
    double r2 = pow(pti.x - ptj.x,2.0) + pow(pti.y - ptj.y,2.0) + pow(pti.z - ptj.z,2.0);
    double dist = sqrt(r2);
    return dist;
}

// find the nearest bound particle to free particle i0
// return its index
int nearest_bound_particle(struct reb_simulation* const r, int i0){
    int j0=-1;
    double mindist=1.0e10;
    for(int j=0; j<r->N-npert;j++){
       double dist = dist_ij(r,i0,j);
       if (dist<mindist){
          mindist=dist;
          j0 = j;
       }
    }
    return j0;
}

// connect free particle with index N-npert to the spring network
// add spring any two particles closer than mush_distance
void connect_free_particle(struct reb_simulation* const r, struct spring spring_vals, 
   double mush_distance){
    int jb = r->N - npert;
    // the index of the particle we want to bind is jb 
    for (int i =0;i< r->N-npert;i++){
       double dist = dist_ij(r, i, jb);
       if (dist < mush_distance){
          add_spring_i(r,i, jb, spring_vals); // will not be added if there is already
                                               // one there
              // spring added at rest distance     , NS will be increased
       }

    } 
    update_reb_springs(r); // update NS in display
    npert --;
}

// loop over all free particles 
// only connect a free particle if it gets within b_distance*bfac of a bound particle
void connect_free_close(struct reb_simulation* const r, double bfac, double mush_fac){
   struct spring spring_mush; // spring parameters for mush
   spring_mush = springs[0]; // use properties of first spring
   double b_distance = r->particles[0].r *2.0; // min dist
   double mush_distance = mush_fac*b_distance;
   int N= r->N;
   for (int i=N-npert; i < N-1;i ++){ // loop over all free particles, but not last one
     int j0 = nearest_bound_particle(r, i); // find nearest  bound  particle
     if (j0<0) return;
     double dist = dist_ij(r,j0,i);        // distance to nearest bound particle
     if (dist < b_distance*bfac){ // we want to connect particle i to network
        printf("%d close\n", i);
        struct reb_particle pti = r->particles[i];  // particle we want to bind
        int j = r->N-npert; // first free particle
        struct reb_particle ptj = r->particles[j];  
        // swap them if they are not the same particle
        if (j!= i){ 
           r->particles[j] =  pti;
           r->particles[i] =  ptj;
        }
        // printf("swapped\n");
        // they are now swapped and j=N-npert is the index of particle that we want to connect
        connect_free_particle(r,spring_mush,mush_distance);
        // printf("connected\n");
        // only connect one particle at a time because we have rearranged the free particle list
        return;
     }
   }
}

