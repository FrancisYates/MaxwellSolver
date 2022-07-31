#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"

const double c = 299792458; // Speed of light
const double mu = 4.0 * M_PI * 1.0e-7; // permiability of free space
const double eps = 1.0 / (c * c * mu); // permitivitty of free space

// must be less than 1/sqrt(2) in 2D
//const double cfl = 0.9 / sqrt(2);
const double cfl = 0.6363961031;

// Grid size in metres
double lengthX;
double lengthY;

// Discretisation in cells
int X;
int Y;

// dx, dy, dt constants.
double dx;
double dy;
double dt = 0.0;

// Time to run for / or number of steps
double T = 0.0001;
int steps = 0;

// Grids used for Yee grid computation
int Ex_size_x, Ex_size_y;
double ** Ex;
int Ey_size_x, Ey_size_y;
double ** Ey;
int Bz_size_x, Bz_size_y;
double ** Bz;

// Resolved grids
int E_size_x, E_size_y, E_size_z;
double *** E;
int B_size_x, B_size_y, B_size_z;
double *** B;

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 * 
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double** A 2D array
 */
double **alloc_2d_array(int m, int n) {
  	double **x;
  	int i;

  	x = (double **)malloc(m*sizeof(double *));
  	x[0] = (double *)calloc(m*n,sizeof(double));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

/**
 * @brief Free a 2D array
 * 
 * @param array The 2D array to free
 */
void free_2d_array(double ** array) {
	free(array[0]);
	free(array);
}

/**
 * @brief Allocate a 3D array that is addressable using square brackets
 * 
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @param o The third dimension of the array
 * @return double*** A 3D array
 */
double ***alloc_3d_array(int m, int n, int o) {
	double ***x;
	x = (double***) malloc(m*sizeof(double **));
	x[0] = (double **) malloc(m*n*sizeof(double *));
	x[0][0] = (double *) calloc(m*n*o,sizeof(double));
	for (int i = 1; i < m; i++) {
		x[i] = &x[0][i*n];
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i == 0 && j == 0) continue;
			x[i][j] = &x[0][0][i*n*o + j*o];
		}
	}
	return x;
}

/**
 * @brief Free a 3D array
 * 
 * @param array The 3D array to free
 */
void free_3d_array(double*** array) {
	free(array[0][0]);
	free(array[0]);
	free(array);
}
