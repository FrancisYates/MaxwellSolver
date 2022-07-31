#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include "args.h"
#include "data.h"
#include "vtk.h"

int verbose = 0;
int input_problem = 0;
int no_output = 0;
int output_freq = 100;
int enable_checkpoints = 0;

static struct option long_options[] = {
    {"cellx",      required_argument, 0, 'x'},
	{"celly",      required_argument, 0, 'y'},
	{"iters",      required_argument, 0, 'n'},
	{"freq",       required_argument, 0, 'f'},
	{"endtime",    required_argument, 0, 't'},
	{"noio",       no_argument,       0, 'd'},
	{"output",     required_argument, 0, 'o'},
	{"checkpoint", no_argument,       0, 'c'},
	{"verbose",    no_argument,       0, 'v'},
	{"help",       no_argument,       0, 'h'},
	{0, 0, 0, 0}
};

/**
 * @brief Print a help message
 * 
 * @param progname The name of the current application
 */
void print_help(char *progname) {
	printf("A simple 2D solver for Maxwell's equations, using Yee method\n");
	printf("Usage: %s [options]\n", progname);
	printf("Options and arguments:\n");
	printf("  -x N, --cellx=N         Cells in X-dimension\n");
	printf("  -y N, --celly=N         Cells in Y-dimension\n");
	printf("  -n N, --iters=N         Iterations (only one of n and t should be specified)\n");
	printf("  -t N, --endtime=N       Set the end time (see -n)\n");
	printf("  -f N, --freq=N          Output frequency (i.e. steps between output)\n");
	printf("  -d, --noio              Disable file I/O\n");
	printf("  -o FILE, --output=FILE  Set base filename for output (final output will be in BASENAME.vtk\n");
	printf("  -c, --checkpoint        Enable checkpointing, checkpoints will be in BASENAME-ITERATION.vtk\n");
	printf("  -v, --verbose           Set verbose output\n");
	printf("  -h, --help              Print this message and exit\n");
	printf("\n");
	printf("Report bugs to <steven.wright@york.ac.uk>\n");
}

/**
 * @brief Parse the argv arguments passed to the application
 * 
 * @param argc The number of arguments present
 * @param argv An array of the arguments presented
 */
void parse_args(int argc, char *argv[]) {
    int option_index = 0;
    char c;

	// You can only specify one of these two arguments
	int n_specified = 0;
	int t_specified = 0;

	while ((c = getopt_long(argc, argv, "x:y:n:t:f:do:cvh", long_options, &option_index)) != -1) {
		switch (c) {
			case 'x':
				X = atoi(optarg);
				break;
			case 'y':
				Y = atoi(optarg);
				break;
			case 'n':
				n_specified = 1;
				steps = atoi(optarg);
				break;
			case 't':
				t_specified = 1;
                T = atof(optarg);
				break;
			case 'f':
				output_freq = atoi(optarg);
				break;
			case 'd':
				no_output = 1;
				break;
			case 'o':
				set_basename(optarg);
				break;
			case 'c':
				enable_checkpoints = 1;
				break;
			case 'v':
				verbose = 1;
				break;
			case '?':
            case 'h':
				print_help(argv[0]);
				exit(1);
		}
	}
	if (n_specified == 1 && t_specified == 1) {
		printf("Error: specify one of -n and -t\n");
		print_help(argv[0]);
		exit(1);
	}
}

/**
 * @brief Print out the current parameters
 * 
 */
void print_opts() {
    printf("=======================================\n");
    printf("Started with the following options\n");
    printf("=======================================\n");
    printf("  lengthX            = %14lf\n", lengthX);
    printf("  lengthY            = %14lf\n", lengthY);
	printf("  steps              = %14d\n", steps);
	printf("  output_freq        = %14d\n", output_freq);
    printf("  T                  = %14.12lf\n", T);
	printf("  no_output          = %14d\n", no_output);
	printf("  enable_checkpoints = %14d\n", enable_checkpoints);
	printf("  basename           = %s\n", get_basename());
    printf("  verbose            = %14d\n", verbose);
    printf("=======================================\n");
    //exit(1);
}