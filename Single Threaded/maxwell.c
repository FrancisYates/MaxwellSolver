#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields() {
	for (int i = 0; i < Bz_size_x; i++) {
		for (int j = 0; j < Bz_size_y; j++) {
			Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
				                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
		}
	}

	for (int i = 0; i < Ex_size_x; i++) {
		for (int j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
		}
	}

	for (int i = 1; i < Ey_size_x-1; i++) {
		for (int j = 0; j < Ey_size_y; j++) {
			Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
		}
	}
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary() {
	for (int i = 0; i < Ex_size_x; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	for (int j = 0; j < Ey_size_y; j++) {
		Ey[0][j] = -Ey[1][j];
		Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	}
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void resolve_to_grid(double *E_mag, double *B_mag) {
	*E_mag = 0.0;
	*B_mag = 0.0;

	for (int i = 1; i < E_size_x-1; i++) {
		for (int j = 1; j < E_size_y-1; j++) {
			E[i][j][0] = (Ex[i-1][j] + Ex[i][j]) / 2.0;
			E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
			//E[i][j][2] = 0.0; // in 2D we don't care about this dimension

			*E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
			//printf("i:%d j:%d E_mag: %1.12f\n",i,j,sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1])));
		}
	}
	
	for (int i = 1; i < B_size_x-1; i++) {
		for (int j = 1; j < B_size_y-1; j++) {
			//B[i][j][0] = 0.0; // in 2D we don't care about these dimensions
			//B[i][j][1] = 0.0;
			B[i][j][2] = (Bz[i-1][j] + Bz[i][j] + Bz[i][j-1] + Bz[i-1][j-1]) / 4.0;

			*B_mag += sqrt(B[i][j][2] * B[i][j][2]);
		}
	}
}

double get_time(){
	struct timeval t;
 	gettimeofday(&t, NULL);
	return (t.tv_sec + (1e-6 * t.tv_usec));
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	double start_time = get_time();
	double elapsed_time;
	
	set_defaults();
	parse_args(argc, argv);
	setup();

	printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
	
	if (verbose) print_opts();
	
	allocate_arrays();

	problem_set_up();

			for (int x=0;x<6;x++){
				for (int y=0;y<6;y++){
					//printf("x:%d y:%d Ex:%14.8e\n",x,y,Ex[x][y]);
				}
			}
	double calc_time = get_time();
	// start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		apply_boundary();
		update_fields();

		t += dt;

		if (i % output_freq == 0) {
			double E_mag, B_mag;
			for (int x=0;x<16;x++){
				for (int y=0;y<3;y++){
					//printf("x:%d y:%d Bz:%14.8e\n",x,y,Bz[x][y]);
				}
			}
			for (int x=24;x<26;x++){
				for (int y=0;y<2;y++){
					//printf("x:%d y:%d Bz:%14.8e\n",x,y,Bz[x][y]);
					//printf("x:%d y:%d Ex:%14.8e\n",x,y,Ex[x][y]);
					//printf("x:%d y:%d Ey:%14.8e\n",x,y,Ey[x][y]);
				}
			}
			resolve_to_grid(&E_mag, &B_mag);
			printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

			if ((!no_output) && (enable_checkpoints))
				write_checkpoint(i);
		}

		i++;
	}

	double E_mag, B_mag;
			for (int x=0;x<16;x++){
				for (int y=0;y<3;y++){
					//printf("x:%d y:%d Bz:%14.8e\n",x,y,Bz[x][y]);
				}
			}
	resolve_to_grid(&E_mag, &B_mag);

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");

	elapsed_time = get_time() - start_time;

	printf("Elapsed Time: %f\n", elapsed_time);
	printf("calculation Time: %f\n", (get_time()-calc_time));

	if (!no_output) 
		write_result();

	free_arrays();

	exit(0);
}


