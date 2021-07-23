

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
double t_datadump;   // for restart data dump outputs 
char froot[30];   // for output files
char restart_froot[30];   // for output files to be read in if a restart takes place

double itaua,itaue; // inverse of migration timescales
double itmig;  // inverse timescale to get rid of migration
double alphaz_1;  // spin drift rates for body 1,2
double alphaz_2;

double *x0_arr,*y0_arr,*z0_arr; // for storing in initial particle positions


void heartbeat(struct reb_simulation* const r);
void del_ends();
void store_xyz0();

int il1,ih1,il2,ih2;  // indices of the two resolved bodies, start and end
static int firstb=0;  // now global for restarts


void additional_forces(struct reb_simulation* r){
   spring_forces(r); // spring forces

}


int main(int argc, char* argv[]){
   struct reb_simulation* const r = reb_create_simulation();
   struct spring spring_mush_1; // spring parameters for mush body 1
   struct spring spring_mush_2; // spring parameters for mush body 2
   // Setup constants
   r->integrator	= REB_INTEGRATOR_LEAPFROG;
   r->gravity		= REB_GRAVITY_BASIC;
   r->boundary		= REB_BOUNDARY_NONE;
   r->G 		= 1;		
   r->additional_forces = additional_forces;  // setup callback function for additional forces

// things to set! can be read in with parameter file
   double tmax = 0.0;  // if 0 integrate forever
   double dt; 

   double mball_1,r_Vol_1;        // total mass, vol equiv radius of ball
   double mball_2,r_Vol_2;        // total mass, vol equiv radius of ball

   double b_distance_1,omegaz_1,ks_1,mush_fac_1;  // for springs
   double b_distance_2,omegaz_2,ks_2,mush_fac_2;
   double gamma_1,gamma_2; // spring damping
   double ratio1_1,ratio2_1,obliquity_deg_1;   // shape and obliquity
   double ratio1_2,ratio2_2,obliquity_deg_2;
        
   double aa,ee,ii,longnode,argperi,meananom; // orbital elements
   unsigned int seed=1;  // for random number generator

// restart types: on command line after  parameter file
// -r  continue integration
// -s  use files given by  restart_froot for masses and springs
//  all other parms are from current parm file

   if (argc ==1){  // no parameter file on command line
        strcpy(froot,"t1");   // to make output files
	dt	   = 1e-3;    // Timestep
        tmax       = 0.0;     // max integration time
        t_print    = 1.0;     // printouts for table
        t_datadump = 1.0e11;  // printouts for data dumps for restarts
        gamma_fac   = 5.0;    // factor initial gamma is higher than regulur gammas
        t_damp      = 1.0;    // gamma from initial gamma 
        seed = 1;  // seed if 1 then use time of day
        strcpy(restart_froot,"r1");   // if a restart from another run  this is the froot of it

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
   // argc != 1
   else{ // read in parameter file which is given on command line
        FILE *fpi;
        fpi = fopen(argv[1],"r");
        char line[300];
        fgets(line,300,fpi);  sscanf(line,"%s",froot);   // fileroot for outputs
        fgets(line,300,fpi);  // globals here!
        fgets(line,300,fpi);  sscanf(line,"%lf",&dt);    // timestep
        fgets(line,300,fpi);  sscanf(line,"%lf",&tmax);  // integrate to this time
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_print); // output timestep
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_datadump); // timestep for mass/spring outputs
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_fac);  // factor initial gamma is higher
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_damp);     // time to switch
        fgets(line,300,fpi);  sscanf(line,"%u",&seed); // random number seed, if 1 then use time of day      
        fgets(line,300,fpi);  sscanf(line,"%s",restart_froot); //restart froot
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

     

   r->dt		= dt; // set integration timestep
   const double boxsize = 2.0*aa;    // display
   reb_configure_box(r,boxsize,1,1,1);
   r->softening      	= b_distance_1/100.0;	// Gravitational softening length

/// end parameters of things to set from parm file read in /////////////////////////

   if (argc == 2){   // we are not doing a restart 
       double obliquity_1 = obliquity_deg_1*M_PI/180.0;   // in radians
       double obliquity_2 = obliquity_deg_2*M_PI/180.0;   // in radians

       if (seed != 1)  // seed random number generator
          srand(seed);

       // properties of springs
       spring_mush_1.gamma     = gamma_fac*gamma_1; // initial damping coefficient
       spring_mush_1.ks        = ks_1; // spring constant
       spring_mush_2.gamma     = gamma_fac*gamma_2; // initial damping coefficient
       spring_mush_2.ks        = ks_2; // spring constant

       // distances for connecting and reconnecting springs
       double mush_distance_1=b_distance_1*mush_fac_1; 
       double mush_distance_2=b_distance_2*mush_fac_2; 

       // to store information about the run, create a file
       FILE *fpr;
       char fname[200];
       sprintf(fname,"%s_run.txt",froot);
       fpr = fopen(fname,"w");

       NS=0; // start with no springs 

       // I input bodies in terms of volume equiv radius 
       // total volume is a*b*c = r_Vol**3  (neglect 4pi/3)
       // a_body**3*ratio1*ratio2 = r_Vol**3
       // a_body = r_Vol*(ratio1*ratio2)**-1/3
       double a_body_1 = r_Vol_1 * pow(ratio1_1 * ratio2_1,-1.0/3.0);
       double a_body_2 = r_Vol_2 * pow(ratio1_2 * ratio2_2,-1.0/3.0);


       // create resolved body particle distributions
       rand_football(r,b_distance_1,a_body_1,a_body_1*ratio1_1, a_body_1*ratio2_1,mball_1);
       il1=0;  // index range for resolved body 1
       ih1=r->N;
       //  this should ignore previously generated particles
       rand_football(r,b_distance_2,a_body_2,a_body_2*ratio1_2, a_body_2*ratio2_2,mball_2);
       il2=ih1; // index range for resolved body 2
       ih2=r->N;

       // record il1,ih1,il2,ih2 indice ranges for body particles
       fprintf(fpr,"il1 ih1 %d %d\n",il1,ih1); 
       fprintf(fpr,"il2 ih2 %d %d\n",il2,ih2); 

       // record desired axis values for bodies 
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

       subtractcom(r,il1,ih1);  // move body 1 to origin 
       subtractcom(r,il2,ih2);  // move body 2 to origin 

       // moments of inertia (eigenvalues)
       double I1_a,I2_a,I3_a;
       double I1_b,I2_b,I3_b;
       // rotate to principal axes
       rotate_to_principal(r, il1, ih1, &I1_a,&I2_a,&I3_a); 
       fprintf(fpr,"I1 I2 I3 (primary) %.6f %.6f %.6f\n", I1_a,I2_a,I3_a);
       rotate_to_principal(r, il2, ih2, &I1_b,&I2_b,&I3_b); 
       fprintf(fpr,"I1 I2 I3 (secondary) %.6f %.6f %.6f\n", I1_b,I2_b,I3_b);
      

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
       int NS_1 = NS; // numbers of springs in first body
       connect_springs_dist(r,mush_distance_2, il2, ih2, spring_mush_2);
       int NS_2 = NS-NS_1; // numbers of springs in second body

       // put the two resolved bodies in a binary orbit   
       // finally we separate them
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
   }

   // we are doing a restart!
   if (argc == 3){
      t_damp = 0;  // no initial damping allowed if a restart
      gamma_fac = 1.0;
      // get the type of restart
      char arg_rs[20];
      strcpy(arg_rs,argv[2]);
      char rtype = arg_rs[1]; // restart type!
      // printf("%c\n",rtype);

      char froot_in[30];
      if (rtype == 's') {
        printf("restart type transfer \n");
        strcpy(froot_in,restart_froot);  // needs to know which froot was used to generate info
      }
      else {
        printf("restart type continuing \n");
        strcpy(froot_in,froot); // same parm file name
        firstb=1;  // make sure only to append to ext files
      }
      read_particles(r,froot_in, 0); // read in particles, sets time
      read_springs(r,froot_in, 0);  // read in springs

      // get the indices for the two extended bodies in the _run file
      FILE *fpi;
      char fname[200];
      sprintf(fname,"%s_run.txt",froot_in);
      fpi = fopen(fname,"r");
      char istring[180],junk1[20],junk2[20];
      fgets(istring,300,fpi);
      sscanf(istring,"%s %s %d %d",junk1,junk2,&il1,&ih1);
      fgets(istring,300,fpi);
      sscanf(istring,"%s %s %d %d",junk1,junk2,&il2,&ih2);
      // printf("%d %d %d %d\n",il1,ih1,il2,ih2);


      if (r->N ==0){
         printf("exit no particles\n");
         exit(0);
      }
      if (NS  ==0){
         printf("exit no springs\n");
         exit(0);
      }

      if (rtype == 's') {
      // on transfer restart set initial time to 0
        r->t=0;
      }
      else{ // continuing restart
        char extendedfile_1[50];
        char extendedfile_2[50];
        sprintf(extendedfile_1,"%s_ext_1.txt",froot);
        sprintf(extendedfile_2,"%s_ext_2.txt",froot);
        del_ends(r,extendedfile_1);  // delete the ends of the ext files
        del_ends(r,extendedfile_2);  // past the time of restart
        // we will get a repeat on the last time in most cases
      }
      // exit(0);
   }

/////////////////////////////// now set up the run

   reb_springs(r); // pass spring index list to display
   r->heartbeat = heartbeat;
   // centerbody(r,il1,ih1);  // move reference frame to resolved body 
   centerbody(r,0,r->N);  // move reference frame to center of mass 
   store_xyz0(r); // store initial conditions

   // max integration time
   if (tmax ==0.0)
      reb_integrate(r, INFINITY);
   else
      reb_integrate(r, tmax);
}


#define NSPACE 50
void heartbeat(struct reb_simulation* const r){
        static int first=0;
        // static int firstb=0; // now global for restarts
        static char extendedfile_1[50];
        static char extendedfile_2[50];
        static char covarfile_1[50];
        static char covarfile_2[50];

        if (first==0){
           first=1;
           sprintf(extendedfile_1,"%s_ext_1.txt",froot);
           sprintf(extendedfile_2,"%s_ext_2.txt",froot);
           sprintf(covarfile_1,"%s_cov_1.txt",froot);
           sprintf(covarfile_2,"%s_cov_2.txt",froot);
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

         
         // table outputs
	 if (reb_output_check(r,t_print)) {
            if (firstb==0){
               firstb=1;
               // first time this outine is called please set last parameter to 1
               // this creates the file and labels on on first line
               // only do this if there is no continued restart!
               print_extended(r,il1,ih1,extendedfile_1,1); // orbital info and stuff
               print_extended(r,il2,ih2,extendedfile_2,1); 
               print_covar(r,il1,ih1, x0_arr, y0_arr, z0_arr, covarfile_1,1);
               print_covar(r,il2,ih2, x0_arr, y0_arr, z0_arr, covarfile_2,1);
            }
            else{ 
               // only append to file
               print_extended(r,il1,ih1,extendedfile_1,0); // orbital info and stuff
               print_extended(r,il2,ih2,extendedfile_2,0); 
               print_covar(r,il1,ih1, x0_arr, y0_arr, z0_arr, covarfile_1,0);
               print_covar(r,il2,ih2, x0_arr, y0_arr, z0_arr, covarfile_2,0);
            }
         }
         // saves for potential restart
	 if (reb_output_check(r,t_datadump)) {
            write_springs(r,froot, 0);
            write_particles(r,froot, 0);
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


// delete the last parts of an ext table files past time of restart
// for a restart continuing, not needed for a restart transfer
void del_ends(struct reb_simulation* const r, char *extfile){

    // copy the extfile to temp.txt
    char sys_command[100];
    sprintf(sys_command,"cp %s temp.txt",extfile);
    printf("%s\n",sys_command);
    system(sys_command);

    FILE *fpi,*fpo;
    char line[400];
    fpi = fopen("temp.txt","r");
    fpo = fopen(extfile,"w");

    fgets(line,400,fpi);  
    fprintf(fpo,"%s",line);
    while(fgets(line,400,fpi) != NULL){
       double tt;
       sscanf(line,"%lf",&tt);
       if (tt <= r->t){
         fprintf(fpo,"%s",line);
       }
    }
    fclose(fpi);
    fclose(fpo);
    
}


// x0_arr,y0_arr,z0_arr are globals
// store initial positions
void store_xyz0(struct reb_simulation* const r){
   x0_arr = malloc(r->N*sizeof(double));
   y0_arr = malloc(r->N*sizeof(double));
   z0_arr = malloc(r->N*sizeof(double));
   for (int i=0;i<r->N;i++){
     x0_arr[i] = r->particles[i].x;
     y0_arr[i] = r->particles[i].y;
     z0_arr[i] = r->particles[i].z;
   }
}
   

