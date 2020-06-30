#pragma once

#ifndef NOMAD_BODY_H_
#define NOMAD_BODY_H_

#include "nomad_option.h"
#include "pool.h"
#include "msg_type.h"
#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>

#include <string>
#include <fstream>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <thread>

using std::vector;
using std::string;
using std::atomic;
//using tbb::atomic;

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
	int max_row_index;
	int local_num_rows;
	long long local_num_nonzero;
};

class NomadBody{

protected:
	bool load_train(const std::string& path,
		int part_index, int num_parts, Data& data);

	bool load_test(const std::string& path,
		int part_index, int num_parts, Data& data);

	// data members:
private:
	int mpi_size, mpi_rank;
	string hostname;

	NomadOption* option;

	int num_parts;
	int global_num_cols;
	vector<Data, callocator<Data>> dstrain;
	vector<Data, callocator<Data>> dstest;
	// summary for train dataset
	vector<int> local_thread_num_rows;
	int global_num_rows;
	vector<long long> local_thread_num_nonzero;
	long long global_num_nonzero;
	std::mt19937_64 rng;

	// create a column pool with big enough size
	// this serves as a memory pool.
	Pool* column_pool;

	// setup initial queues of columns
	// each thread owns each queue with corresponding access
	colque* job_queues;
	// a queue of columns to be sent to other machines via network
	colque send_queue;
	// a queue of special signal, messages in this queue is sent even when allow_sending==false
	tbb::concurrent_queue<std::pair<int, int>> send_queue_force;
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

	// array used to remember the sizes of send_queue in each machine
	atomic<int>* queue_current_sizes;
	// we try to bound the size of send_queue's by this number
	int queue_upperbound;	//const

	// updater_func related
	atomic<int> wait_number;
	// these arrays will be used to calculate test error
	// each thread will calculate test error on its own, and the results will be aggregated
	int* train_count_errors;
	double* train_sum_errors;
	int* test_count_errors;
	double* test_sum_errors;

	// define constants needed for network communication
	int column_per_msg;
	// col_index + vector
	int unit_bytenum;	//const
	// current queue size + number of columns + columns
	const int msg_head_bytenum = sizeof(int8_t) + sizeof(int) + sizeof(int);	//const
	int msg_bytenum;	//const

	long long local_send_count = 0;

	// data for checkpoint
	// THE MAIN WORK FOR CHECKPOINT IS DONE IN updater_func() FOR MINIMIZEING THE MODIFICATION
	atomic<bool> finished;
	atomic<bool> checkpointing;
	atomic<int> cp_ut_wait_counter;
	atomic<bool>* cp_action_ready; // each thread
	atomic<int> cp_received_clear_counter;
	vector<atomic<bool>*> cp_need_archive_msg_from; // each thread - mpi instance
	atomic<int>* cp_need_archive_msg_counter; // each thread

	//vector<int, callocator<int> > count_recv_flush;
	//vector<vector<bool>, callocator<vector<bool> > > received_flush;
	int cp_epoch;

	vector<long long, callocator<long long> > msg_archived;
	vector<double, callocator<double> > cp_time_write;
	double cp_time_total_worker;
	tbb::tick_count cp_time_total_timer;
	double cp_time_total_master;
	vector<std::ofstream*> cp_fmsgs;

	// master - checkpoint
	int cp_master_epoch;
	std::mutex cp_m;
	std::condition_variable cp_cv;
	atomic<int> cp_master_lfinish_count;

	// master - termination check
	int tm_count;
	double global_error;
	std::mutex tm_m;
	std::condition_variable tm_cv;
	vector<double, callocator<double>> tm_local_error_received;
	atomic<bool>* tm_local_error_ready;
	int tm_min_updated_col;
	// number of updated column on each mpi instance
	vector<long long, callocator<double>> tm_local_update_count;
	long long tm_global_update_count;

	// worker - termination check (online RMSE)
	// local sum error for each column
	vector<double, callocator<double>> tm_col_error;
	// incrementally updated in recv_thread and send_thread
	double tm_col_error_sum;

	// network control
	bool control_net_delay;
	std::chrono::duration<double> net_delay;
	bool control_net_ratio;
	double net_ratio;

	// others
	string log_header;
	atomic<bool> allow_sending;
	atomic<bool> allow_processing;
	atomic<bool>* allow_processing_thread;

	// private functions:
private:
	bool initial_mpi();
	int get_num_cols(const std::string& path);
	void initial_dataset();
	void initial_data4thread();
	void initial_data4machine();
	void initial_net_data();
	void initial_termcheck();
	void initial_cp();
	void initial_net_control();
	bool initial(NomadOption* opt);

	void do_net_control_delay();
	void do_net_control_ratio(size_t nbyte, tbb::tick_count::interval_t time);

	/////////////////////////////////////////////////////////
	// Define Master Thread
	/////////////////////////////////////////////////////////
	void master_checkpoint();
	void master_termcheck();
	void sh_m_lerror(int source, double error, long long count);
	void cp_sh_m_lfinish(int source_part);

	/////////////////////////////////////////////////////////
	// Define Updater Thread
	/////////////////////////////////////////////////////////
	void updater_func(int thread_index);

	/////////////////////////////////////////////////////////
	// Define Message Sending/Broadcasting Functions
	/////////////////////////////////////////////////////////
	void _send_data(char* send_message, const int cur_num, const int target_rank);
	void _send_lerror(const double lerror, const long long nupdate);
	void _bcast_dying();
	void _bcast_termination(const int epoch);
	void _bcast_cp_start(const int epoch);
	void _bcast_cp_clear(const int epoch);
	void _send_cp_lfinish(const int epoch);
	void _bcast_cp_resume(const int epoch);

	/////////////////////////////////////////////////////////
	// Define Training Threads
	/////////////////////////////////////////////////////////
	void train_send_func(const double timeout);
	void train_recv_func();

	/////////////////////////////////////////////////////////
	// Define Testing Threads
	/////////////////////////////////////////////////////////
	void test_send_func();
	void test_recv_func();

	/////////////////////////////////////////////////////////
	// Define Checkpoint functions
	/////////////////////////////////////////////////////////
	void _send_clear_signal(bool direct_send);
	void _send_lfinish_signal();
	string gen_cp_file_name(int part_index);

	//bool start_cp(int thread_index, int epoch);
	void archive_local(int thread_index, double* latent_rows, int local_num_rows);
	void _archive_msg_queue(int thread_index, const string& suffix, colque& queue, bool locked = true);
	void arhive_job_queue(int thread_index, bool locked = true);
	void arhive_send_queue(bool locked = true);
	void archive_msg(int thread_index, ColumnData* p_col);
	//void finish_cp(int thread_index);

	void restore_local(const string& cp_f, int part_index, double* latent_rows, int& local_num_rows, int& dim);
	void restore_msg_queue(const string& cp_f, colque& queue);
	void restore(int epoch, int thread_index, double* latent_rows, int& local_num_rows, int& dim);

	void _send_sig2threads_start(int epoch);
	void _send_sig2threads_clear(int source); // source rank
	void _send_sig2threads_resume(int epoch);
	void _sync_all_update_thread();

	// signal handler - machine (MPI instance) level
	void cp_shm_start(int epoch);
	void cp_shm_clear(int epoch, int source); // source rank
	void cp_shm_resume(int epoch);

	// signal handler - thread level
	void cp_sht_start(int thread_index, int part_index, int epoch, double* latent_rows, int local_num_rows);
	void cp_sht_clear(int thread_index, int part_index, int source, double* latent_rows, int local_num_rows);
	void cp_sht_resume(int thread_index, int part_index, int epoch);

	// cp action which only execute once
	void cp_update_func_action(int thread_index, int part_index, double* latent_rows, int local_num_rows);

public:
	int run(NomadOption* opt);

};

#endif
