/**
 * @file        spring.h
 * @brief       springs 
 * @author      Alice Quilen 
 */


#ifndef _SPRING_H
#define _SPRING_H


#include"particle.h"

// spring structure 
struct spring  {
	double ks;   // spring constant
	double rs0;  // distance of no force
	double gamma; // damping coefficient
	double k_heat;  //  heat diffusion coefficient!
	int    i;    // vertex 1 referring to a particle
	int    j;    // vertex 2 referring to a particle
};

// coordination structure for each node
struct coord {
       int ncoord;  // number of springs connected to this node
       int coordlist[50];  // a list of spring indices ** note number here is hard wired, giving a max!
       double lx;
       double ly;
       double lz;
};

struct stresstensor {  // for each node
        double sigxx; //stress tensor
        double sigyy;
        double sigzz;
        double sigxy;
        double sigyz;
        double sigxz;
        double eig1;  // eigenvalues of stress tensor eig1 is biggest
        double eig2;
        double eig3;
        double maxF;  // max Force of a spring
        int s_index;  // index of spring giving max Force
        int fail;  // is there material failure 
};

struct node {
        int surf;  // is 1 if is a surface node
        double temp;  // temperature
        double cv;    // specific heat, probably integrated over mass of node?
};

struct symtensor {
       double xx;
       double yy;
       double zz;
       double xy;
       double yz;
       double xz;
};


extern struct spring* springs;
extern int NS; // numbers of springs
extern int NPERT; // number of point masses
extern double b_distance; // mush formation
extern double mush_distance; // mush spring connection 
extern double t_reform; // formation springs timescale
extern double gamma_all; // for gamma  of all springs

void spring_forces();
double spring_potential_energy();
double grav_potential_energy();

// list of springs related subroutines
struct node *mknodevec();
void surface_nodes();
void nfilename();
void print_node();
void transport_heat();
void heat_nodes_tidal();
void heat_nodes_radiogenic();
double Kthermal_mush();
void adjust_spring_temp_lin();
void adjust_spring_temp_ts();
int add_spring_i();
void del_spring();
void spring_force_k_one();
void zero_symtensor();

void print_stress();
void update_stresstensor();
int markfailure();
void sfilename();
void killsprings();

int nearest_to_shape();
void print_surf();
int surface_shape(); //ZYH
void surfaceparticle_display(); //ZYH
void potoang(); //ZYH
int *marksurface();
int *marksurface_football();
int *marksurface_cone();
void rescale_xyz();
double min_radius();
double max_radius();
void rand_bennu();
void rmvertices();
void rmshape_vertices();
void read_vertex_file();
void adjust_ks_abc();
void adjust_ks_abc_fac();
void adjust_mass_abc();
void subtractcov();
void subtractcom();
void print_extended();
void print_covar();
void print_extended_simp();
void print_extended_2nodes();
void print_pm();
void hfilename();
void quadJ2pole();
void quadC22S22();
void quadforce();
double add_pt_mass_kep();
void move_resolved();
double sum_mass();
double dEdt();
double dEdt_total();
void spr_xyz_mid();
void compute_com();
void compute_cov();
void compute_Lorb();
double spring_length();
double strain();
void springs_add();
void normalize ();
double mindist();
void centerbody();
void connect_springs_dist();
void connect_springs_dist_p();
void connect_springs_dist_nodemax();
void rand_football();
void rand_rectangle();
void rand_rectangle_2d();
void rand_cone();
void rand_football_from_sphere();
double Young_mush();
double Young_mush_big();
void set_gamma();
void set_gamma_fac();
void spin();
void make_binary_spring();
void mom_inertia();
void measure_L();
void measure_L_origin();
void compute_semi();
void compute_semi_bin();
void total_mom();
double mean_L();
void spring_init(struct reb_simulation* r);
void output_png();
void output_png_single();
void spr_ang_mid();
void body_spin();
void print_tab();
void print_bin();
void print_heat();
void read_particles();
void write_particles();
void write_springs();
void read_springs();
void toistring();
void zero_accel();
void invI();
double detI();
void eigenvalues();
void eigenvecs();
void rotate_to_principal();
void vec_mul();
void adjust_ks();
void adjust_mass_side();
void rotate_body();
void rotate_origin();
void rotate_vector();
double fill_hcp();
double fill_cubic();
double add_pluto_charon();
double add_pluto_charon_kep();
// double add_one_mass();
double add_one_mass_kep();
void add_one_mass_cartesian();
void dodrift_bin();
void dodrift_res();
void dodrift_twores();
void addto_heatvec();
void norm_heatvec();
double compute_rot_kin();
void adjust_nodes_cp();
void stretch();
void orbels_twobodies();
double add_twores_bin();
void drift_spin();


#endif // _SPRING_H


