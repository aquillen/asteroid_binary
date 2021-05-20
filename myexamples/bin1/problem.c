

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

double gamma_fac; // for setting gamma of all springs
double t_damp;    // end faster damping, relaxation
double t_print;   // for table printout 
char froot[30];   // output files

double itaua,itaue; // inverse of migration timescales
double itmig;  // inverse timescale to get rid of migration
double alphaz_1;  // spin drift rates for body 1,2
double alphaz_2;

void heartbeat(struct reb_simulation* const r);

int il1,ih1,il2,ih2;


void additional_forces(struct reb_simulation* r){
   spring_forces(r); // spring forces

}


int main(int argc, char* argv[]){
	struct reb_simulation* const r = reb_create_simulation();
        struct spring spring_mush_1; // spring parameters for mush body 1
        struct spring spring_mush_2; // spring parameters for mush body 2
	// Setup constants
	r->integrator	= REB_INTEGRATOR_LEAPFROG;
	r->gravity	= REB_GRAVITY_BASIC;
	r->boundary	= REB_BOUNDARY_NONE;
	r->G 		= 1;		
        r->additional_forces = additional_forces;  // setup callback function for additional forces
        // int lattice_type = 0; // random lattice

// things to set! can be read in with parameter file
        double tmax = 0.0;  // if 0 integrate forever
        double dt; 

        double mball_1,r_Vol_1;        // total mass, radius of ball
        double mball_2,r_Vol_2;        // total mass, radius of ball

        double b_distance_1,omegaz_1,ks_1,mush_fac_1;
        double b_distance_2,omegaz_2,ks_2,mush_fac_2;
        double gamma_1,gamma_2;
        double ratio1_1,ratio2_1,obliquity_deg_1;
        double ratio1_2,ratio2_2,obliquity_deg_2;
        
        double aa,ee,ii,longnode,argperi,meananom;
        unsigned int seed=1;

    if (argc ==1){
        strcpy(froot,"t1");   // to make output files
	dt	   = 1e-3;    // Timestep
        tmax       = 0.0;     // max integration time
        t_print    = 1.0;     // printouts for table
        gamma_fac   = 5.0;    // factor initial gamma is higher than regulur gammas
        t_damp      = 1.0;    // gamma from initial gamma 
        seed = 1;  // seed if 1 then use time of day
        // drifts
        itaua    = 0.0;   // inverse drift rate in a
        itaue    = 0.0;   // inverse drift rate in e
        itmig    = 0.0;   // get rid of drift rate in inverse of this time
	// orbit
        aa       =3.0;    // semi-major axis orbital
	ee       =0.0;    // initial eccentricity
	ii       =0.0;    // initial inclination 
	longnode =0.0;    // initial longnode
	argperi  =0.0;    // initial argument of pericenter
	meananom =0.0;    // initial mean anomali

        // resolved body 1
        mball_1 = 1.0;        // mass 
        r_Vol_1 = 1.0;        // volume equiv radius
        b_distance_1 = 0.25;  // min separation between particles
        mush_fac_1  = 2.3;    // ratio of smallest spring distance to minimum interparticle dist
        ks_1        = 8e-2;   // spring constant
        // spring damping
        gamma_1     = 1.0;    // final damping coeff
        ratio1_1    = 1.0;    // axis ratio resolved body   b/a
        ratio2_1    = 1.0;    // axis ratio c/a
        omegaz_1    = 0.3;    // initial spin
        obliquity_deg_1= 0.0; // obliquity
        alphaz_1 = 0.0;       // spin drift rate  z-axis
        // resolved body 2
        mball_2 = 0.1;        // mass 
        r_Vol_2 = 0.46;       // volume equiv radius
        b_distance_2 = 0.15;  // min separation between particles
        mush_fac_2  = 2.3;    // ratio of smallest spring distance to minimum interparticle dist
        ks_2        = 8e-2;   // spring constant
        // spring damping
        gamma_2     = 1.0;    // final damping coeff
        ratio1_2    = 1.0;    // axis ratio resolved body   b/a
        ratio2_2    = 1.0;    // axis ratio c/a
        omegaz_2    = 0.1;    // initial spin
        obliquity_deg_2= 0.0; // obliquity
        alphaz_2 = 0.0;      // spin drift rate 

     }
     else{
        FILE *fpi;
        fpi = fopen(argv[1],"r");
        char line[300];
        fgets(line,300,fpi);  sscanf(line,"%s",froot);   // fileroot for outputs
        fgets(line,300,fpi);  // globals here!
        fgets(line,300,fpi);  sscanf(line,"%lf",&dt);    // timestep
        fgets(line,300,fpi);  sscanf(line,"%lf",&tmax);  // integrate to this time
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_print); // output timestep
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_fac);  // factor initial gamma is higher
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_damp);     // time to switch
        fgets(line,300,fpi);  sscanf(line,"%u",&seed); // random number seed, if 1 then use time of day      
        fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf",&itaua,&itaue,&itmig);  // drifts
        fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf %lf %lf %lf",
             &aa,&ee,&ii,&longnode,&argperi,&meananom); // orbital elements

        fgets(line,300,fpi);  // resolved body 1
        fgets(line,300,fpi);  sscanf(line,"%lf",&mball_1); // mass 
        fgets(line,300,fpi);  sscanf(line,"%lf",&r_Vol_1); // volume equiv radius
        fgets(line,300,fpi);  sscanf(line,"%lf",&b_distance_1); // min interparticle distance
        fgets(line,300,fpi);  sscanf(line,"%lf",&mush_fac_1);   // sets max spring length
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks_1);         // spring constant
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_1);  // damping final
        fgets(line,300,fpi);  sscanf(line,"%lf %lf",&ratio1_1,&ratio2_1); 
                 // axis ratios for body b/a, c/a
        fgets(line,300,fpi);  sscanf(line,"%lf",&omegaz_1);     // initial body spin
        fgets(line,300,fpi);  sscanf(line,"%lf",&obliquity_deg_1); // obliquity degrees
        fgets(line,300,fpi);  sscanf(line,"%lf",&alphaz_1); // spin drift rate z axis

        fgets(line,300,fpi);  // resolved body 2
        fgets(line,300,fpi);  sscanf(line,"%lf",&mball_2); // mass 
        fgets(line,300,fpi);  sscanf(line,"%lf",&r_Vol_2); // volume equiv radius
        fgets(line,300,fpi);  sscanf(line,"%lf",&b_distance_2); // min interparticle distance
        fgets(line,300,fpi);  sscanf(line,"%lf",&mush_fac_2);   // sets max spring length
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks_2);         // spring constant
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_2);  // damping final
        fgets(line,300,fpi);  sscanf(line,"%lf %lf",&ratio1_2,&ratio2_2); 
        fgets(line,300,fpi);  sscanf(line,"%lf",&omegaz_2);     // initial body spin
        fgets(line,300,fpi);  sscanf(line,"%lf",&obliquity_deg_2); // obliquity degrees
        fgets(line,300,fpi);  sscanf(line,"%lf",&alphaz_2); // spin drift
     }

     
     double obliquity_1 = obliquity_deg_1*M_PI/180.0;   // in radians
     double obliquity_2 = obliquity_deg_2*M_PI/180.0;   // in radians


/// end parameters of things to set /////////////////////////

     r->dt=dt; // set integration timestep
     const double boxsize = 2.0*aa;    // display
     reb_configure_box(r,boxsize,1,1,1);
     r->softening      = b_distance_1/100.0;	// Gravitational softening length


   // properties of springs
   spring_mush_1.gamma     = gamma_fac*gamma_1; // initial damping coefficient
   spring_mush_1.ks        = ks_1; // spring constant
   spring_mush_2.gamma     = gamma_fac*gamma_2; // initial damping coefficient
   spring_mush_2.ks        = ks_2; // spring constant

   double mush_distance_1=b_distance_1*mush_fac_1; 
   double mush_distance_2=b_distance_2*mush_fac_2; 
       // distance for connecting and reconnecting springs

   
   // to store information about the run
   FILE *fpr;
   char fname[200];
   sprintf(fname,"%s_run.txt",froot);
   fpr = fopen(fname,"w");

   NS=0; // start with no springs 

// I input in terms of volume equi radius 
   // total volume is a*b*c = r_Vol**3  (neglect 4pi/3)
   // a_body**3*ratio1*ratio2 = r_Vol**3
   // a_body = r_Vol*(ratio1*ratio2)**-1/3
   double a_body_1 = r_Vol_1 * pow(ratio1_1 * ratio2_1,-1.0/3.0);
   double a_body_2 = r_Vol_2 * pow(ratio1_2 * ratio2_2,-1.0/3.0);

   // record axis values for bodies 
   fprintf(fpr,"a_1 %.3f\n",a_body_1); 
   fprintf(fpr,"b_1 %.3f\n",a_body_1*ratio1_1); 
   fprintf(fpr,"c_1 %.3f\n",a_body_1*ratio2_1); 
   // double volume_ratio_1 = pow(a_body_1,3.0)*ratio1_1*ratio2_1;  // neglecting 4pi/3 factor
   // fprintf(fpr,"vol_ratio_1 %.6f\n",volume_ratio_1); // with respect to 4pi/3 
   // so I can check that it is set to 1
   fprintf(fpr,"a_2 %.3f\n",a_body_2); 
   fprintf(fpr,"b_2 %.3f\n",a_body_2*ratio1_2); 
   fprintf(fpr,"c_2 %.3f\n",a_body_2*ratio2_2); 
   // double volume_ratio_2 = pow(a_body_1,3.0)*ratio1_1*ratio2_1;  // neglecting 4pi/3 factor
   // fprintf(fpr,"vol_ratio_2 %.6f\n",volume_ratio_2); // with respect to 4pi/3 

   if (seed != 1)  // seed random number generator
      srand(seed);

   // create resolved body particle distributions
   rand_football(r,b_distance_1,a_body_1,a_body_1*ratio1_1, a_body_1*ratio2_1,mball_1);
   il1=0;  // index range for resolved body
   ih1=r->N;
   //  this should ignore previously generated particles
   rand_football(r,b_distance_2,a_body_2,a_body_2*ratio1_2, a_body_2*ratio2_2,mball_2);
   il2=ih1;
   ih2=r->N;

   subtractcom(r,il1,ih1);  // move body 1 to origin 
   subtractcom(r,il2,ih2);  // move body 2 to origin 

   rotate_to_principal(r, il1, ih1); // rotate to principal axes
   rotate_to_principal(r, il2, ih2); 
   // rotate_body(r, il, ih, alpha, beta, ggamma);

// spin the two bodies  
   subtractcov(r,il1,ih1); // center of velocity subtracted 
   subtractcov(r,il2,ih2); 
   spin(r,il1, ih1, 0.0, 0.0, omegaz_1);  // change one of these zeros to tilt it!
   spin(r,il2, ih2, 0.0, 0.0, omegaz_2); 
   subtractcov(r,il1,ih1); // center of velocity subtracted 
   subtractcov(r,il2,ih2); 
   double speriod_1  = fabs(2.0*M_PI/omegaz_1);
   double speriod_2  = fabs(2.0*M_PI/omegaz_2);
   printf("spin period body 1 %.6f\n",speriod_1);
   printf("spin period body 2 %.6f\n",speriod_2);
   fprintf(fpr,"spin period body 1 %.6f\n",speriod_1);
   fprintf(fpr,"spin period body 2 %.6f\n",speriod_2);
   rotate_body(r, il1, ih1, 0.0, obliquity_1, 0.0); // tilt by obliquity
   rotate_body(r, il2, ih2, 0.0, obliquity_2, 0.0); // 

// make springs, all pairs connected within interparticle distance mush_distance
   connect_springs_dist(r,mush_distance_1, il1, ih1, spring_mush_1);
   int NS_1 = NS;
   connect_springs_dist(r,mush_distance_2, il2, ih2, spring_mush_2);
   int NS_2 = NS-NS_1;

// put the two resolved bodies in a binary orbit   
   double nn = add_twores_bin(r, il1, ih1, il2, ih2, aa,ee, ii, longnode,argperi,meananom);
   // nn is the mean motion of the binary orbit
   double orb_period  = fabs(2.0*M_PI/nn);
   printf("orbital period  %.6f\n",orb_period);
   fprintf(fpr,"orbital period  %.6f\n",orb_period);
   printf("mean motion  %.6f\n",nn);
   fprintf(fpr,"mean motion  %.6f\n",nn);

   // ratio of numbers of particles to numbers of springs for resolved body
   int N_1 = ih1 - il1;
   int N_2 = ih2 - il2;
   double Nratio_1 = (double)NS_1/(double)N_1;
   double Nratio_2 = (double)NS_2/(double)N_2;
   printf("N_1=%d NS_1=%d NS_1/N_1=%.1f\n", N_1, NS_1, Nratio_1);
   printf("N_2=%d NS_2=%d NS_2/N_2=%.1f\n", N_2, NS_2, Nratio_2);
   fprintf(fpr,"N_1 %d\n", N_1);
   fprintf(fpr,"N_2 %d\n", N_2);
   fprintf(fpr,"NS_1 %d\n",  NS_1);
   fprintf(fpr,"NS_2 %d\n",  NS_2);
   fprintf(fpr,"NS_1/N_1 %.1f\n", Nratio_1);
   fprintf(fpr,"NS_2/N_2 %.1f\n", Nratio_2);

// compute elatic modulus for each body
   double Emush1 = Young_mush(r,il1,ih1, 0.0, r_Vol_1/2);
   double Emush2 = Young_mush(r,il2,ih2, 0.0, r_Vol_2/2);
   printf("Young's modulus body 1 %.2f\n",Emush1);
   printf("Young's modulus body 2 %.2f\n",Emush2);
   fprintf(fpr,"Young's modulus body 1 %.2f\n",Emush1);
   fprintf(fpr,"Young's modulus body 2 %.2f\n",Emush2);


   fclose(fpr);

   reb_springs(r); // pass spring index list to display
   r->heartbeat = heartbeat;
   // centerbody(r,il1,ih1);  // move reference frame to resolved body 
   centerbody(r,0,r->N);  // move reference frame to center of mass 

   // max integration time
   if (tmax ==0.0)
      reb_integrate(r, INFINITY);
   else
      reb_integrate(r, tmax);
}


#define NSPACE 50
void heartbeat(struct reb_simulation* const r){
        static int first=0;
        static int firstb=0;
        static char extendedfile_1[50];
        static char extendedfile_2[50];

        if (first==0){
           first=1;
           sprintf(extendedfile_1,"%s_ext_1.txt",froot);
           sprintf(extendedfile_2,"%s_ext_2.txt",froot);
        }
	if (reb_output_check(r,10.0*r->dt)){
		reb_output_timing(r,0);
	}
        if (fabs(r->t - t_damp) < 0.9*r->dt) set_gamma_fac(gamma_fac,0,r->N); 
            // damp initial bounce only 
            // reset gamma only at t near t_damp
	
         // stuff to do every timestep
         centerbody(r,0,r->N);  // move reference frame to center of mass for display
         // do orbital migration
         double migfac = exp(-1.0* r->t*itmig);  
         dodrift_twores(r, r->dt,itaua*migfac, itaue*migfac, il1, ih1, il2, ih2);
         // drift spin of both bodies
         drift_spin(r, r->dt, il1, ih1, 0.0, 0.0, alphaz_1);
         drift_spin(r, r->dt, il2, ih2, 0.0, 0.0, alphaz_2);

         
	 if (reb_output_check(r,t_print)) {
            if (firstb==0){
               firstb=1;
               print_extended(r,il1,ih1,extendedfile_1,1); // orbital info and stuff
               print_extended(r,il2,ih2,extendedfile_2,1); 
            }
            else{ 
               print_extended(r,il1,ih1,extendedfile_1,0); // orbital info and stuff
               print_extended(r,il2,ih2,extendedfile_2,0); 
            }
         }




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





