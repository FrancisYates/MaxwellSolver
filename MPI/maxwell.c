#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/time.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"


double get_time(){
	struct timeval t;
 	gettimeofday(&t, NULL);
	return (t.tv_sec + (1e-6 * t.tv_usec));
}

void find_field_iters(int* i_srt, int* i_lim, int rank, int num_process, int id){
	int split;
	if (id == 0){
		split = ceil((float)(Bz_size_x)/(float)num_process);
		*i_srt = rank * split;
		if (rank<num_process-1){
			*i_lim  = (rank+1)*split;
		}else{
			*i_lim = Bz_size_x;
		}
	}else if (id == 1){
		split = ceil((float)(Ex_size_x)/(float)num_process);
		*i_srt = rank * split;
		if (rank<num_process-1){
			*i_lim  = (rank+1)*split;
		}else{
			*i_lim = Ex_size_x;
		}
	}else if (id == 2){
		split = ceil((float)(Bz_size_x)/(float)num_process);
		*i_srt = rank == 0? rank*split+1 : rank*split;
		if (rank<num_process-1){
			*i_lim  = (rank+1)*split;
		}else{
			*i_lim = Ey_size_x - 1;
		}	
	}else if (id == 3){
		split = ceil((float)(Ex_size_x)/(float)num_process);
		*i_srt = rank * split;
		if (rank < num_process-1){
			*i_lim = (rank + 1) * split;
		}else{
			*i_lim = Ex_size_x;
		}
	}else if (id == 4){
		split = ceil((float)(Ex_size_y)/(float)num_process);
		*i_srt = rank * split;
		if (rank < num_process-1){
			*i_lim = (rank + 1) * split;
		}else{
			*i_lim = Ex_size_y;
		}
	}else if (id == 5){
		split = ceil((float)(E_size_x)/(float)num_process);
		*i_srt = rank == 0? rank*split+1 : rank*split;
		if (rank<num_process-1){
			*i_lim  = (rank+1)*split;
		}else{
			*i_lim = E_size_x-1;
		}
	}else{
		split = ceil((float)(Bz_size_x)/(float)num_process);
		*i_srt = rank == 0? rank*split+1 : rank*split;
		if (rank<num_process-1){
			*i_lim  = (rank+1)*split;
		}else{
			*i_lim = B_size_x-1;
		}	
	}
}

int get_y_from_id(int id){
	if(id == 0){
		return Bz_size_y;
	}else if(id == 1){
		return Ex_size_y;
	}else if(id == 2){
		return Ey_size_y;
	}else if(id == 3){
		return Ex_size_y;
	}else if(id == 4)
		return Ey_size_y;
}

void send_recieve_field(int rank, int num_process, double **field, int call_id, int msg_id, int send_first, int depth){
	int num_items, i_srt, i_lim;
	int field_y = get_y_from_id(msg_id);
	MPI_Status status;

	if(send_first == 1){
		find_field_iters(&i_srt, &i_lim, rank, num_process, msg_id);
		num_items = (i_lim - i_srt) * field_y;
		MPI_Send((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, call_id, msg_id, MPI_COMM_WORLD);		
		//printf("rank:%d depth%d here1\n", rank, depth);

		find_field_iters(&i_srt, &i_lim, call_id, num_process, msg_id);
		num_items = (i_lim - i_srt) * field_y;
		MPI_Recv((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, call_id, msg_id, MPI_COMM_WORLD, &status);
		//printf("rank:%d depth%d here2\n", rank, depth);
	}else{
		find_field_iters(&i_srt, &i_lim, call_id, num_process, msg_id);
		num_items = (i_lim - i_srt) * field_y;
		MPI_Recv((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, call_id, msg_id, MPI_COMM_WORLD, &status);
		//printf("rank:%d depth%d here1\n", rank, depth);
		find_field_iters(&i_srt, &i_lim, rank, num_process, msg_id);
		num_items = (i_lim - i_srt) * field_y;
		MPI_Send((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, call_id, msg_id, MPI_COMM_WORLD);
		//printf("rank:%d depth%d here2\n", rank, depth);
	}
}

void split_array(int *procsL, int *procsR, int *even, int *all_Procs, int procs_len,int rank, int depth){
	*even = ((procs_len % 2) == 0) ? 1 : 0;
	int itr_lim = *even == 1 ? procs_len: procs_len-1;
	itr_lim = procs_len;
	int arr_len = procs_len/2;
	
	for(int i=0; i<arr_len; i++){
		procsL[i] = all_Procs[i];
		procsR[i]  = all_Procs[i+arr_len];
	}
}

void rotate_array(int *array, int arr_len){
	int end_val = array[arr_len-1];
	for(int i=(arr_len-1); i>0; i--){
		array[i]=array[i-1];
	}
	array[0] = end_val;
}

int is_in_array(int rank, int arr_len, int *array,int depth){
	for (int i=0; i<arr_len; i++){
		if(rank == array[i]){
			return 1;
		}
	}
	return 0;
}

int index_in_array(int rank, int arr_len, int *array){
	for (int i=0; i<arr_len; i++){
		if(rank == array[i]){
			return i;
		}
	}
}

void recursive(int rank, int num_process, int num_procs, int *procs, double **field, int id, int *unfinished, int depth){
	int my_idx, field_y;
	int even_procs;
	int arr_len = num_procs/2;
	int *procsL = calloc(arr_len, sizeof(int));
	int *procsR = calloc(arr_len, sizeof(int));
	depth+=1;
	split_array(procsL, procsR, &even_procs, procs, num_procs,rank,depth);
	int in_L_arr = is_in_array(rank, arr_len, procsL,depth);
	if(in_L_arr == 1){
		my_idx = index_in_array(rank, arr_len, procsL);
	}else{
		my_idx = index_in_array(rank, arr_len, procsR);
	}
	

	for(int i=0; i<arr_len; i++){
		int partner_rank;
		int check_val = ceil((float)num_procs/(float)2);
		if(in_L_arr == 1){
			partner_rank = procsR[my_idx];
			send_recieve_field(rank, num_process, field, partner_rank, id, 1,depth);
		}else if(is_in_array(rank, arr_len, procsR,depth) == 1){
			partner_rank = procsL[my_idx];
			send_recieve_field(rank, num_process, field, partner_rank, id, 0, depth);
		}
		
		rotate_array(procsR, arr_len);
		if(in_L_arr != 1){
			my_idx = (my_idx+1 == arr_len) ? 0 : my_idx+1;
		}
	}

	if (arr_len>=2){
		//printf("rank:%d depth%d rec\n", rank, depth);
		recursive(rank, num_process, arr_len, procsL, field, id, unfinished, depth);
		recursive(rank, num_process, arr_len, procsR, field, id, unfinished, depth);
	}

	if (even_procs == 0){
		unfinished[procs[num_procs-1]] = 1;
		if (rank==0){
			//printf("i:%d un:%d\n",num_procs-1, procs[num_procs-1]);
		}
	}
}

void send_field_to_odd_proc(int rank, int num_process, double **field, int odd_rank, int msg_id){
	int num_items, i_srt, i_lim;
	int field_y = get_y_from_id(msg_id);
	MPI_Status status;

	if (rank == odd_rank){
		for (int i =0;i<num_process;i++){
			if(rank!=i){		
				find_field_iters(&i_srt, &i_lim, i, num_process, msg_id);
				num_items = (i_lim - i_srt) * field_y;
				MPI_Recv((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, i, msg_id, MPI_COMM_WORLD, &status);

			}
		}
	}else{
		find_field_iters(&i_srt, &i_lim, rank, num_process, msg_id);
		num_items = (i_lim - i_srt) * field_y;
		MPI_Send((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, odd_rank, msg_id, MPI_COMM_WORLD);
	}

}

void recursive_send_recieve_field(int rank, int num_process, double **field, int id){
	trans_time_start = get_time();
	int arr_len = num_process/2;
	int *procs = calloc(num_process, sizeof(int));
	int *unfinished = calloc(num_process, sizeof(int));
	
	for(int i=0; i<num_process; i++){
		procs[i] = i;
		unfinished[i] = 0;
	}

	recursive(rank, num_process, num_process, procs, field, id, unfinished,0);
	//printf("rank:%d here1\n", rank);

	int num_items, field_y;	
	int i_srt, i_lim;
		if (rank==0){
			//printf("loop\n");
		}
	for(int i=0; i<num_process; i++){
		if (rank==0){
			//printf("i:%d un:%d\n",i, unfinished[i]);
		}
		if(unfinished[i] == 1){
			field_y = get_y_from_id(id);
			find_field_iters(&i_srt, &i_lim, i, num_process, id);
			num_items = (i_lim - i_srt) * field_y;
			MPI_Bcast(&(field[i_srt][0]), num_items, MPI_DOUBLE, i, MPI_COMM_WORLD);
			send_field_to_odd_proc(rank,num_process,field, i,id);
		}
	}
	trans_time += get_time() - trans_time_start;

}

void send_to_right(int rank, int num_process, double **field, int id){
	trans_time_start = get_time();
	int num_items, i_srt, i_lim, send_to, recv_frm;
	int field_y = get_y_from_id(id);
	int *idx;
	MPI_Status status;

	send_to = rank+1;
	recv_frm = rank-1;

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank % 2 ==0){
		if(send_to < num_process){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1 * field_y;
			MPI_Send((void*)&(field[i_lim-1][0]), num_items, MPI_DOUBLE, send_to, id, MPI_COMM_WORLD);
		}

		if(recv_frm >= 0){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Recv((void*)&(field[i_srt-1][0]), num_items, MPI_DOUBLE, recv_frm, id, MPI_COMM_WORLD, &status);
		}
	}else{
		if( recv_frm >= 0){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Recv((void*)&(field[i_srt-1][0]), num_items, MPI_DOUBLE, recv_frm, id, MPI_COMM_WORLD, &status);
		}

		if(send_to < num_process){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Send((void*)&(field[i_lim-1][0]), num_items, MPI_DOUBLE, send_to, id, MPI_COMM_WORLD);
		}
	}
	trans_time += get_time() - trans_time_start;
}

void send_to_left(int rank, int num_process, double **field, int id){
	trans_time_start = get_time();
	int num_items, i_srt, i_lim, send_to, recv_frm;
	int field_y = get_y_from_id(id);
	MPI_Status status;

	send_to = rank-1;
	recv_frm = rank+1;

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank % 2 ==0){
		if(send_to >= 0){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1 * field_y;
			MPI_Send((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, send_to, id, MPI_COMM_WORLD);
		}

		if(recv_frm < num_process){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Recv((void*)&(field[i_lim][0]), num_items, MPI_DOUBLE, recv_frm, id, MPI_COMM_WORLD, &status);
		}
	}else{
		if( recv_frm < num_process){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Recv((void*)&(field[i_lim][0]), num_items, MPI_DOUBLE, recv_frm, id, MPI_COMM_WORLD, &status);
		}

		if(send_to >= 0){
			find_field_iters(&i_srt, &i_lim, rank, num_process, id);
			num_items = 1  * field_y;
			MPI_Send((void*)&(field[i_srt][0]), num_items, MPI_DOUBLE, send_to, id, MPI_COMM_WORLD);
		}
	}
	trans_time += get_time() - trans_time_start;
}

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void MPI_update_fields(int rank, int num_process) {
	int i_srt;
	int i_lim;

	find_field_iters(&i_srt, &i_lim, rank, num_process, 0);
	for (int i = i_srt; i < i_lim; i++) {
		for (int j = 0; j < Bz_size_y; j++) {
			Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
				                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	send_to_right(rank, num_process, Bz, 0);
	
	find_field_iters(&i_srt, &i_lim, rank, num_process, 1);
	for (int i = i_srt; i < i_lim; i++) {
		for (int j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
		}
	}
	find_field_iters(&i_srt, &i_lim, rank, num_process, 2);
	for (int i = i_srt; i < i_lim; i++) {
		for (int j = 0; j < Ey_size_y; j++) {
			Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
		}
	}
	
	send_to_left(rank, num_process, Ey, 2);
}

/**
 * @brief Apply boundary conditions
 * 
 */
void MPI_apply_boundary(int rank, int num_process) {
	int i_srt = rank * Ex_size_x;
	int i_lim;
	int j_srt = rank * Ey_size_y;
	int j_lim;
	int num_items;
	MPI_Status status;
	
	find_field_iters(&i_srt, &i_lim, rank, num_process, 3);
	find_field_iters(&j_srt, &j_lim, rank, num_process, 4);

	for (int i = i_srt; i < i_lim; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	for (int j = j_srt; j < j_lim; j++) {
		Ey[0][j] = -Ey[1][j];
		Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	}
	
	recursive_send_recieve_field(rank, num_process, Ex, 3);
	//MPI_Barrier(MPI_COMM_WORLD);
	recursive_send_recieve_field(rank, num_process, Ey, 4);
	//MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void MPI_resolve_to_grid(double *E_mag, double *B_mag, int rank, int num_process) {
	int i_srt;
	int i_lim;
	double temp_E_mag, temp_B_mag;
	MPI_Status status;
	*E_mag = 0.0;
	*B_mag = 0.0;
		
	find_field_iters(&i_srt, &i_lim, rank, num_process, 5);

	for (int i = i_srt; i < i_lim; i++) {
		for (int j = 1; j < E_size_y-1; j++) {
			E[i][j][0] = (Ex[i-1][j] + Ex[i][j]) / 2.0;
			E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
			//E[i][j][2] = 0.0; // in 2D we don't care about this dimension

			*E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
		}
	}
	
	find_field_iters(&i_srt, &i_lim, rank, num_process, 6);
	for (int i = i_srt; i < i_lim; i++) {
		for (int j = 1; j < B_size_y-1; j++) {
			//B[i][j][0] = 0.0; // in 2D we don't care about these dimensions
			//B[i][j][1] = 0.0;
			B[i][j][2] = (Bz[i-1][j] + Bz[i][j] + Bz[i][j-1] + Bz[i-1][j-1]) / 4.0;

			*B_mag += sqrt(B[i][j][2] * B[i][j][2]);
		}
	}
	
	if (rank != 0){
		MPI_Send((void*)E_mag, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
		MPI_Send((void*)B_mag, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
	}else{
		MPI_Status status;
		for(int i=1; i<num_process; i++){
			MPI_Recv((void*)&temp_E_mag, 1, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, &status);
			MPI_Recv((void*)&temp_B_mag, 1, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, &status);
			*E_mag += temp_E_mag;
			*B_mag += temp_B_mag;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

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
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	double total_time_start, setup_time_start, field_time_start, resolve_time_start, calculation_time_start, post_time_start;
	double total_time, setup_time, field_time, resolve_time, calculation_time, post_time;
	int num_process;
	int rank;

	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("rank:%d  process:%d\n", rank, num_process);

	total_time_start = get_time();
	setup_time_start = get_time();
	set_defaults();
	parse_args(argc, argv);
	setup();

	if (rank == 0){
		printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
		
		if (verbose) print_opts();
	}

	allocate_arrays();

	problem_set_up();

	setup_time = get_time() - setup_time_start;
	calculation_time_start = get_time();

	field_time = 0.0;
	resolve_time = 0.0;
	// start at time 0
	double t = 0.0; 
	int i = 0;
	while (i < steps) {
		field_time_start = get_time();
		apply_boundary();
		MPI_update_fields(rank, num_process);
		field_time += get_time() - field_time_start;

		t += dt;
		
		if (i % output_freq == 0) {
			double E_mag, B_mag;
			recursive_send_recieve_field(rank, num_process, Bz, 0);
			recursive_send_recieve_field(rank, num_process, Ex,1);
			recursive_send_recieve_field(rank, num_process, Ey, 2);

			MPI_resolve_to_grid(&E_mag, &B_mag, rank, num_process);

			if (rank == 0){
				printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);				
			}
			resolve_time += get_time() - resolve_time_start;

			//MPI_Barrier(MPI_COMM_WORLD);

			if ((!no_output) && (enable_checkpoints))
				write_checkpoint(i);
		}

		i++;
	}

	double E_mag, B_mag;
	resolve_time_start = get_time();
	recursive_send_recieve_field(rank, num_process, Bz, 0);
	recursive_send_recieve_field(rank, num_process, Ex,1);
	recursive_send_recieve_field(rank, num_process, Ey, 2);
	
	MPI_resolve_to_grid(&E_mag, &B_mag, rank, num_process);
	resolve_time += get_time() - resolve_time_start;
	calculation_time = (get_time() - calculation_time_start)-trans_time;
	post_time_start = get_time();
	if (rank == 0){
		printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
		printf("Simulation complete.\n");

		if (!no_output){
			write_result();
		}
	}
	
	free_arrays();
	
	post_time = get_time() - post_time_start;
	total_time = get_time() - total_time_start;
	if (rank == 0){
		printf("Total Time: %f\n", total_time);
		printf("Setup Time: %f\n", setup_time);
		printf("Field Time: %f\n", field_time);
		printf("Reslove Time: %f\n", resolve_time);
		printf("Calculation Time: %f\n", calculation_time);
		printf("Post Time: %f\n", post_time);
		printf("Transmision Time: %f\n", trans_time);
	}
	MPI_Finalize();

	exit(0);
}
