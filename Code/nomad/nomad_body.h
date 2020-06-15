#pragma once

#ifndef NOMAD_BODY_H_
#define NOMAD_BODY_H_

#include "nomad_option.h"
#include "pool.hpp"
#include "msg_type.h"
#include "tbb/tbb.h"

#include <string>
#include <fstream>
#include <vector>
#include <condition_variable>
#include <atomic>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

#include "CheckpointState.h"

#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"

constexpr int UNITS_PER_MSG = 100;

using std::vector;
using std::string;
using std::atomic;

//using tbb::atomic;
using tbb::tick_count;

using nomad::ColumnData;
using nomad::MsgType;

template <typename T>
using sallocator = tbb::scalable_allocator<T>;

template <typename T>
using callocator = tbb::cache_aligned_allocator<T>;

using colque = tbb::concurrent_queue<ColumnData*, callocator<ColumnData*> >;


struct Data{
	std::vector<int, sallocator<int> > col_offset;
	std::vector<int, sallocator<int> > row_idx;
	std::vector<double, sallocator<double> > row_val;
	int num_rows;
	int num_cols;
	long long num_nonzero;

	int min_row_index;
	int local_num_rows;
};

class NomadBody{

protected:
	bool load_train(const std::string& path,
		int part_index, int num_parts, bool show_info,
		Data& data
	);

	bool load_test(const std::string& path,
		int part_index, int num_parts, bool show_info,
		Data& data
	);

	// data members:
private:
	int numtasks, rank;
	int hostname_len;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	nomad::NomadOption* option;

	int num_parts;
	int global_num_cols;
	rng_type rng;

	// create a column pool with big enough size
	// this serves as a memory pool.
	nomad::Pool* column_pool;

	// setup initial queues of columns
	// each thread owns each queue with corresponding access
	colque* job_queues;
	// a queue of columns to be sent to other machines via network
	colque send_queue;
	// save columns here, and push to job_queues again before next train starts
	vector<ColumnData*, sallocator<ColumnData*> > saved_columns;

	// count the number of threads in the machine which initial setup for training is done
	atomic<int> count_setup_threads;
	// this flag will be turned on when all threads are ready for training
	atomic<bool> flag_train_ready;
	// this flag will be used to send signals to all threads that it has to stop training
	atomic<bool> flag_train_stop;
	// this flag will be turned on when all threads are ready for testing
	atomic<bool> flag_test_ready;
	// this flag will be used to send signals to all threads that it has to stop testing
	atomic<bool> flag_test_stop;

	// maintain the number of updates for each thread
	atomic<long long>* num_updates;
	// also maintain a number of pop failures from job queue
	atomic<long long>* num_failures;
	// used to compute the number of empty columns inside a machine
	// BUGBUG: right now it does not have any purpose other than computing statistics
	// we may enhance the efficiency of communication by taking advantage of this information
	atomic<bool>* is_column_empty;

	// array used to remember the sizes of send_queue in each machine
	atomic<int>* queue_current_sizes;
	// we try to bound the size of send_queue's by this number
	int queue_upperbound;	//const

	// updater_func related
	tbb::atomic<int> wait_number;
	// these arrays will be used to calculate test error
	// each thread will calculate test error on its own, and the results will be aggregated
	int* train_count_errors;
	double* train_sum_errors;
	int* test_count_errors;
	double* test_sum_errors;

	// define constants needed for network communication
	// col_index + vector
	int unit_bytenum;	//const
	// current queue size + number of columns + columns
	int msg_bytenum;	//const

	long long local_send_count = 0;

	// data for checkpoint
	// THE MAIN WORK FOR CHECKPOINT IS DONE IN updater_func() FOR MINIMIZEING THE MODIFICATION
	bool finished;
	vector<bool, callocator<bool> > checkpointing;
	vector<int, callocator<int> > count_recv_flush;
	vector<vector<bool>, callocator<vector<bool> > > received_flush;
	vector<int, callocator<int> > cp_epoch;
	//	vector<CheckpointState, callocator<CheckpointState> > cp_state;
	string cp_folder;

	vector<long long, callocator<long long> > msg_archived;
	vector<double, callocator<double> > cp_write_time;

	// for master
	int cp_master_epoch;
	std::mutex cp_m;
	std::condition_variable cp_cv;
	const int cp_signal_start = -1; //initialized once and used multiple times
	const int cp_signal_flush = -2; //initialized once and used multiple times
	vector<std::ofstream*> cp_fmsgs;

	// network control
	bool control_net_delay;
	std::chrono::duration<double> net_delay;
	bool control_net_ratio;
	double net_ratio;

	// others
	string log_header;

	// private functions:
private:
	bool initial_mpi();
	bool initial_option(int argc, char** argv);
	int get_num_cols(const std::string& path);
	void initial_data4thread();
	void initial_data4machine();
	void initial_net_data();
	void initial_cp();
	void initial_net_control();
	bool initial(int argc, char** argv);

	void do_net_control_delay();
	void do_net_control_ratio(size_t nbyte, tbb::tick_count::interval_t time);

	/////////////////////////////////////////////////////////
	// Define Master Thread
	/////////////////////////////////////////////////////////
	void master_func();

	/////////////////////////////////////////////////////////
	// Define Updater Thread
	/////////////////////////////////////////////////////////
	void updater_func(int thread_index);

	/////////////////////////////////////////////////////////
	// Define Training Sender Thread
	/////////////////////////////////////////////////////////
	void _send_msg(char* send_message, const int cur_num, const int target_rank);
	void train_send_func(const double timeout);

	/////////////////////////////////////////////////////////
	// Define Training Receive Function
	/////////////////////////////////////////////////////////
	void train_recv_func();

	/////////////////////////////////////////////////////////
	// Define Testing Sender Thread
	/////////////////////////////////////////////////////////
	void test_send_func();

	/////////////////////////////////////////////////////////
	// Define Testing Receive Function
	/////////////////////////////////////////////////////////
	void test_recv_func();

	/////////////////////////////////////////////////////////
	// Define Checkpoint functions
	/////////////////////////////////////////////////////////
	void signal_handler_start(int thread_index, ColumnData* p_col, double* latent_rows, int local_num_rows, int dim);
	void signal_handler_flush(int thread_index, ColumnData* p_col);

	bool start_cp(int thread_index, int epoch);
	void archive_local(int thread_index, double* latent_rows, int local_num_rows, int dim);
	void archive_msg(int thread_index, ColumnData* p_col);
	void finish_cp(int thread_index);

	void restore_local(const string& cp_folder, int part_index, double* latent_rows, int& local_num_rows, int& dim);
	void restore_msg(const string& cp_folder, int part_index);
	void restore(int epoch, int thread_index, double* latent_rows, int& local_num_rows, int& dim);

public:
	int run(int argc, char** argv);

};

#endif
