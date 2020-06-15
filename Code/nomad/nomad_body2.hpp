/*
 * Copyright (c) 2016 Tian Zhou
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

#ifndef NOMAD_NOMAD_BODY_HPP_
#define NOMAD_NOMAD_BODY_HPP_

#include "nomad.hpp"
#include "nomad_option.h"
#include "pool.hpp"
#include "msg_type.h"
#include "tbb/tbb.h"
 //#define TBB_IMPLEMENT_CPP0X
//#include <tbb/compat/thread>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <condition_variable>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

#include "CheckpointState.h"

const int UNITS_PER_MSG = 100;

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::ios;

using std::vector;
using std::pair;
using std::string;

using tbb::atomic;
using tbb::tick_count;

using nomad::ColumnData;
using nomad::MsgType;

typedef tbb::concurrent_queue<ColumnData*, callocator<ColumnData*> > colque;

using nomad::NomadOption;

class NomadBody{

protected:
	virtual NomadOption* create_option() = 0;

	virtual bool load_train(NomadOption& option,
		int part_index, int num_parts,
		int& min_row_index,
		int& local_num_rows,
		vector<int, sallocator<int> >& col_offset,
		vector<int, sallocator<int> >& row_idx,
		vector<scalar, sallocator<scalar> >& row_val
	) = 0;

	virtual bool load_test(NomadOption& option,
		int part_index, int num_parts,
		int& min_row_index,
		int& local_num_rows,
		vector<int, sallocator<int> >& col_offset,
		vector<int, sallocator<int> >& row_idx,
		vector<scalar, sallocator<scalar> >& row_val
	) = 0;

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
	real* train_sum_errors;
	int* test_count_errors;
	real* test_sum_errors;

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
	vector<ofstream*, callocator<ofstream*> > cp_fmsgs;

	// network control
	bool control_net_delay;
	std::chrono::duration<double> net_delay;
	bool control_net_ratio;
	double net_ratio;

	// private functions:
private:
	bool initial_mpi(){
		int mpi_thread_provided;
		MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
		if(mpi_thread_provided != MPI_THREAD_MULTIPLE && mpi_thread_provided != MPI_THREAD_SERIALIZED){
			cerr << "MPI multiple thread not provided!!! (" << mpi_thread_provided << " != "
				<< MPI_THREAD_MULTIPLE << " or " << MPI_THREAD_SERIALIZED << ")" << endl;
			return false;
		}

		// retrieve MPI task info
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
		MPI_Get_processor_name(hostname, &hostname_len);
		return true;
	}
	bool initial_option(int argc, char** argv){
		option = create_option();
		return option->parse_command(argc, argv);
	}
	void initial_data4thread(){
		// maintain the number of updates for each thread
		num_updates = callocator<atomic<long long> >().allocate(option->num_threads_);
		for(int i = 0; i < option->num_threads_; i++){
			callocator<atomic<long long> >().construct(num_updates + i);
			num_updates[i] = 0;
		}

		// also maintain a number of pop failures from job queue
		num_failures = callocator<atomic<long long> >().allocate(option->num_threads_);
		for(int i = 0; i < option->num_threads_; i++){
			callocator<atomic<long long> >().construct(num_failures + i);
			num_failures[i] = 0;
		}

		// setup initial queues of columns
		// each thread owns each queue with corresponding access
		job_queues = callocator<colque>().allocate(option->num_threads_);
		for(int i = 0; i < option->num_threads_; i++){
			callocator<colque>().construct(job_queues + i);
		}
	}
	void initial_data4machine(){
		// used to compute the number of empty columns inside a machine
		// BUGBUG: right now it does not have any purpose other than computing statistics
		// we may enhance the efficiency of communication by taking advantage of this information
		is_column_empty = callocator<atomic<bool> >().allocate(global_num_cols);
		for(int i = 0; i < global_num_cols; i++){
			is_column_empty[i] = true;
		}
	}
	void initial_net_data(){
		// array used to remember the sizes of send_queue in each machine
		queue_current_sizes = callocator<atomic<int> >().allocate(numtasks);
		for(int i = 0; i < numtasks; i++){
			queue_current_sizes[i] = 0;
		}
		// we try to bound the size of send_queue's by this number
		queue_upperbound = global_num_cols * 4 / numtasks;

		// define constants needed for network communication
		// col_index + vector
		unit_bytenum = sizeof(int) + sizeof(long) + sizeof(scalar) * option->latent_dimension_;
		// current queue size + number of columns + columns
		msg_bytenum = sizeof(int) + sizeof(int) + unit_bytenum * UNITS_PER_MSG;
	}
	void initial_cp(){
		//master:
		cp_master_epoch = 0;
		//worker:
		cp_epoch.resize(option->num_threads_, -1);
		checkpointing.resize(option->num_threads_, false);
		received_flush.resize(option->num_threads_, vector<bool>(num_parts, false));
		count_recv_flush.resize(option->num_threads_, 0);
		cp_fmsgs.resize(option->num_threads_);
		//		cp_state.resize(option->num_threads_, CheckpointState(num_parts));
		msg_archived.resize(option->num_threads_, 0);
		cp_write_time.resize(option->num_threads_, 0.0);
	}
	void initial_net_control(){
		control_net_delay = option->net_delay > 0;
		if(control_net_delay){
			net_delay = std::chrono::duration<double>(option->net_delay);
		}
		control_net_ratio = option->net_delay != std::numeric_limits<double>::max();
		net_ratio = option->net_ratio;
	}
	bool initial(int argc, char** argv){
		if(!initial_option(argc, argv) || !initial_mpi())
			return false;
		cout << boost::format("processor name: %s, number of tasks: %d, rank: %d") % hostname % numtasks % rank << endl;

		num_parts = numtasks * option->num_threads_;

		cout << "number of threads: " << option->num_threads_ << ", number of parts: " << num_parts << endl;

		// read number of columns
		global_num_cols = option->get_num_cols();

		initial_data4thread();
		initial_data4machine();
		initial_net_data();

		// create a column pool with big enough size
		// this serves as a memory pool. global_num_cols * 3 / num_parts is arbitrary big enough number.
		// when the capacity is exceeded, it automatically assigns additional memory.
		// therefore no need to worry too much
		column_pool = new nomad::Pool(option->latent_dimension_, option->num_threads_,
			std::min(global_num_cols * 3 / num_parts, global_num_cols));

		//for updater_func
		wait_number = 0;

		// distribution used to initialize parameters
		// distribution is taken from Hsiang-Fu's implementation of DSGD
		rng = rng_type(option->seed_ + rank * 131 + 139);

		// these arrays will be used to calculate test error
		// each thread will calculate test error on its own, and the results will be aggregated
		train_count_errors = callocator<int>().allocate(option->num_threads_);
		train_sum_errors = callocator<real>().allocate(option->num_threads_);
		test_count_errors = callocator<int>().allocate(option->num_threads_);
		test_sum_errors = callocator<real>().allocate(option->num_threads_);

		initial_cp();
		initial_net_control();
		finished = false;
		return true;
	}
	void do_net_control_delay(){
		if(control_net_delay > 0){
			std::this_thread::sleep_for(net_delay);
		}
	}
	void do_net_control_ratio(size_t nbyte, tbb::tick_count::interval_t time){
		if(!control_net_ratio)
			return;
		double t = nbyte / net_ratio - time.seconds();
		cout << t << endl;
		if(t > 0){
			std::this_thread::sleep_for(std::chrono::duration<double>(t));
		}
	}
	/////////////////////////////////////////////////////////
	// Define Master Thread
	/////////////////////////////////////////////////////////
	void master_func(){
		cout << "master thread start" << endl;
		std::unique_lock<std::mutex> lk(cp_m);
		tick_count last_cptime = tbb::tick_count::now();
		std::chrono::duration<double> cp_interval(option->cp_interval_);
		while(!finished){
			bool btm = cp_cv.wait_for(lk, cp_interval, [&](){
				return (tbb::tick_count::now() - last_cptime).seconds() >= option->cp_interval_;
				});
			if(!finished && flag_train_ready && !flag_train_stop){
				if(!btm)
					continue;
				cout << "sending out checkpoint signal " << cp_master_epoch << endl;
				for(int i = 0; i < numtasks; ++i){
					MPI_Ssend(reinterpret_cast<char*>(&cp_master_epoch), sizeof(cp_master_epoch), MPI_CHAR, i, MsgType::CP_START, MPI_COMM_WORLD);
				}
				cp_master_epoch++;
				last_cptime = tbb::tick_count::now();
			}
		}
	}


	/////////////////////////////////////////////////////////
	// Define Updater Thread
	/////////////////////////////////////////////////////////
	void updater_func(int thread_index){
		int part_index = rank * option->num_threads_ + thread_index;
		cout << boost::format("rank: %d, thread_index: %d, part_index: %d") % rank % thread_index % part_index << endl;

		/////////////////////////////////////////////////////////
		// Read Data
		/////////////////////////////////////////////////////////

		// each thread reads its own portion of data and stores in CSC format
		vector<int, sallocator<int> > train_col_offset, test_col_offset;
		vector<int, sallocator<int> > train_row_idx, test_row_idx;
		vector<scalar, sallocator<scalar> > train_row_val, test_row_val;

		int local_num_rows;
		int min_row_index;

		bool succeed = load_train(*option, part_index, num_parts, min_row_index, local_num_rows,
			train_col_offset, train_row_idx, train_row_val);
		if(succeed == false){
			cerr << "error in reading training file" << endl;
			exit(11);
		}

		succeed = load_test(*option, part_index, num_parts, min_row_index, local_num_rows,
			test_col_offset, test_row_idx, test_row_val);
		if(succeed == false){
			cerr << "error in reading testing file" << endl;
			exit(11);
		}

		for(int i = 0; i < global_num_cols; i++){
			if(train_col_offset[i + 1] > train_col_offset[i]){
				is_column_empty[i].compare_and_swap(false, true);
			}
		}

		/////////////////////////////////////////////////////////
		// Initialize Data Structure
		/////////////////////////////////////////////////////////

		// now assign parameters for rows
		scalar* latent_rows = sallocator<scalar>().allocate(local_num_rows * option->latent_dimension_);

		// initialize random number generator
		rng_type rng(option->seed_ + rank * 131 + thread_index + 1);
		std::uniform_real_distribution<scalar> init_dist(0, 1.0 / sqrt(option->latent_dimension_));
		for(int i = 0; i < local_num_rows * option->latent_dimension_; i++){
			latent_rows[i] = init_dist(rng);
		}

		int* col_update_counts = sallocator<int>().allocate(global_num_cols);
		std::fill_n(col_update_counts, global_num_cols, 0);

		// copy some essential parameters explicitly

		const int dim = option->latent_dimension_;
		const scalar learn_rate = option->learn_rate_;
		const scalar decay_rate = option->decay_rate_;
		const scalar par_lambda = option->par_lambda_;
		const int num_threads = option->num_threads_;
		const int num_reuse = option->num_reuse_;

		long long local_num_updates = 0;
		long long local_num_failures = 0;

		// notify that the thread is ready to run
		count_setup_threads++;

		for(unsigned int timeout_iter = 0; timeout_iter < option->timeouts_.size(); timeout_iter++){

			cout << "thread: " << thread_index << " ready to train!" << endl;

			// wait until all threads are ready
			while(flag_train_ready == false){
				std::this_thread::yield();
			}

			/////////////////////////////////////////////////////////
			// Training
			/////////////////////////////////////////////////////////

			while(flag_train_stop == false){

				ColumnData* p_col;
				bool pop_succeed = job_queues[thread_index].try_pop(p_col);

				if(pop_succeed){ // there was an available column in job queue to process
					// CP checking:
					if(p_col->col_index_ == cp_signal_start){
						// CP: start
						signal_handler_start(thread_index, p_col, latent_rows, local_num_rows, dim);
						continue;
					} else if(p_col->col_index_ == cp_signal_flush){
						signal_handler_flush(thread_index, p_col);
						column_pool->push(p_col);
						continue;
					}

					// normal process:
					const int col_index = p_col->col_index_;
					const scalar step_size = learn_rate * 1.5
						/ (1.0 + decay_rate * pow(col_update_counts[col_index] + 1, 1.5));

					scalar* col = p_col->values_;

					// for each data point
					for(int offset = train_col_offset[col_index];
						offset < train_col_offset[col_index + 1]; offset++){

						// retrieve the point
						int row_index = train_row_idx[offset];
						scalar* row = latent_rows + dim * row_index;

						// calculate error
						scalar cur_error = std::inner_product(col, col + dim, row, -train_row_val[offset]);

						// update both row and column
						for(int i = 0; i < dim; i++){
							scalar tmp = row[i];

							row[i] -= step_size * (cur_error * col[i] + par_lambda * tmp);
							col[i] -= step_size * (cur_error * tmp + par_lambda * col[i]);
						}

						local_num_updates++;

					}

					col_update_counts[col_index]++;

					// CP: archive message
					if(checkpointing[thread_index] && !received_flush[thread_index][p_col->source_]){
						archive_msg(thread_index, p_col);
					}

					// send to the next thread
					p_col->pos_++;
					// if the column was circulated in every thread inside the machine, send to another machine
					if(p_col->pos_ >= num_threads * num_reuse){
						// BUGBUG: now treating one machine case as special.. should I continue doing this?
						if(numtasks == 1){
							p_col->pos_ = 0;
							p_col->source_ = part_index;
							job_queues[p_col->perm_[p_col->pos_ % num_threads]].push(p_col);
						} else{
							send_queue.push(p_col);
						}
					} else{
						p_col->source_ = part_index;
						job_queues[p_col->perm_[p_col->pos_ % num_threads]].push(p_col);
					}

				} else{
					local_num_failures++;
					std::this_thread::yield();
				}

			}

			num_updates[thread_index] = local_num_updates;
			num_failures[thread_index] = local_num_failures;

			while(flag_test_ready == false){
				std::this_thread::yield();
			}

			/////////////////////////////////////////////////////////
			// Testing
			/////////////////////////////////////////////////////////

			int num_col_processed = 0;

			real train_sum_squared_error = 0.0;
			int train_count_error = 0;

			real test_sum_squared_error = 0.0;
			int test_count_error = 0;

			//int monitor_num = 0;
			//tbb::tick_count start_time = tbb::tick_count::now();

			// test until every column is processed
			while(num_col_processed < global_num_cols){

				//double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
				//if(monitor_num < elapsed_seconds){
				//	cout << "test updater alive," << rank << ","<< monitor_num << ","
				//			<< num_col_processed << "/" << global_num_cols << "" << endl;
				//	monitor_num++;
				//}

				ColumnData* p_col;

				if(job_queues[thread_index].try_pop(p_col)){

					scalar* col = p_col->values_;
					const int col_index = p_col->col_index_;

					// for each training data point
					for(int offset = train_col_offset[col_index];
						offset < train_col_offset[col_index + 1]; offset++){

						// retrieve the point
						int row_index = train_row_idx[offset];
						scalar* row = latent_rows + dim * row_index;

						// calculate error
						scalar cur_error = -train_row_val[offset];
						for(int i = 0; i < dim; i++){
							cur_error += col[i] * row[i];
						}

						train_sum_squared_error += cur_error * cur_error;
						train_count_error++;

					}

					// for each test data point
					for(int offset = test_col_offset[col_index];
						offset < test_col_offset[col_index + 1]; offset++){

						// retrieve the point
						int row_index = test_row_idx[offset];
						scalar* row = latent_rows + dim * row_index;

						// calculate error
						scalar cur_error = -test_row_val[offset];
						for(int i = 0; i < dim; i++){
							cur_error += col[i] * row[i];
						}

						test_sum_squared_error += cur_error * cur_error;
						test_count_error++;

					}

					if(thread_index < num_threads - 1){
						job_queues[thread_index + 1].push(p_col);
					} else{
						send_queue.push(p_col);
					}

					num_col_processed++;

				} else{
					std::this_thread::yield();
				}

			}

			train_count_errors[thread_index] = train_count_error;
			train_sum_errors[thread_index] = train_sum_squared_error;

			test_count_errors[thread_index] = test_count_error;
			test_sum_errors[thread_index] = test_sum_squared_error;

			// notify that this thread has finished testing
			count_setup_threads++;

		}

		// print to the file
		if(option->output_path_.length() > 0){

			while(wait_number < part_index % option->num_threads_){
				std::this_thread::yield();
			}

			ofstream::openmode mode = (part_index % option->num_threads_ == 0) ?
				ofstream::out : (ofstream::out | ofstream::app);
			ofstream ofs(option->output_path_ + std::to_string(rank), mode);

			cout << "min_row_index: " << min_row_index << endl;
			for(int i = 0; i < local_num_rows; i++){
				scalar* row = latent_rows + dim * i;
				ofs << "row," << (min_row_index + i);
				for(int t = 0; t < dim; t++){
					ofs << "," << row[t];
				}
				ofs << endl;
			}
			ofs.close();

			wait_number++;

		}

		sallocator<int>().deallocate(col_update_counts, global_num_cols);

		sallocator<scalar>().deallocate(latent_rows, local_num_rows * option->latent_dimension_);
	}

	/////////////////////////////////////////////////////////
	// Define Training Sender Thread
	/////////////////////////////////////////////////////////
	void _send_msg(char* send_message, const int cur_num, const int target_rank){
		*(reinterpret_cast<int*>(send_message)) = send_queue.unsafe_size();
		*(reinterpret_cast<int*>(send_message) + 1) = cur_num;
		//tbb::tick_count t = tbb::tick_count::now();
		int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);
		if(rc != MPI_SUCCESS){
			std::cerr << "SendTask MPI Error" << std::endl;
			exit(64);
		}

		//do_net_control_ratio(msg_bytenum, tbb::tick_count::now() - t);

		local_send_count += cur_num;
	}
	void train_send_func(const double timeout){

		rng_type send_rng(rank * 17 + option->seed_ + option->num_threads_ + 2);
		std::uniform_int_distribution<> target_dist(0, numtasks - 1);

		const int dim = option->latent_dimension_;

		while(flag_train_ready == false){
			std::this_thread::yield();
		}

		const tick_count start_time = tick_count::now();
		//		int monitor_num = 0;

		char* send_message = sallocator<char>().allocate(msg_bytenum);
		char* cur_pos = send_message + sizeof(int) + sizeof(int);
		int cur_num = 0;

		// Buffer some columns for one message. (# column>UNITS_PER_MSG || wait time>timeout)
		while(true){

			double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
			if(elapsed_seconds > timeout &&
				std::none_of(checkpointing.begin(), checkpointing.end(), [](const bool v){return v; })){
				break;
			}

			//			if(monitor_num < elapsed_seconds){
			//				cout << "sender thread alive," << rank << "," << monitor_num << ","
			//						<< send_queue.unsafe_size() << ",endline" << endl;
			//				monitor_num++;
			//			}

			ColumnData* p_col = nullptr;

			if(send_queue.try_pop(p_col)){
				if(p_col->col_index_ == cp_signal_flush){
					//flush out out-message
					if(cur_num != 0){
						int target_rank = target_dist(send_rng);
						_send_msg(send_message, cur_num, target_rank);

						local_send_count += cur_num;
						cur_pos = send_message + sizeof(int) + sizeof(int);
						cur_num = 0;
					}
					//send flush signal
					*reinterpret_cast<int*>(send_message) = p_col->pos_;	//set source part_index
					for(int target_rank = 0; target_rank < numtasks; ++target_rank){
						if(target_rank == rank)
							continue;
						int rc = MPI_Ssend(send_message, sizeof(int), MPI_CHAR, target_rank, MsgType::CP_FLUSH, MPI_COMM_WORLD);
						if(rc != MPI_SUCCESS){
							std::cerr << "SendTask MPI Error" << std::endl;
							exit(64);
						}
					}
					column_pool->push(p_col);
					continue;
				}

				p_col->serialize(cur_pos, dim);
				column_pool->push(p_col);

				cur_pos += unit_bytenum;
				cur_num++;

				if(cur_num >= UNITS_PER_MSG){

					int target_rank = target_dist(send_rng);
					_send_msg(send_message, cur_num, target_rank);

					local_send_count += cur_num;
					cur_pos = send_message + sizeof(int) + sizeof(int);
					cur_num = 0;

					// choose destination
//					while(true){
//						int target_rank = target_dist(send_rng);
//						if(queue_current_sizes[target_rank] < queue_upperbound){
//
//							int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA,MPI_COMM_WORLD);
//							// BUGBUG: in rank 0, arbitrary delay is added
//							if(rank == 0 && option->rank0_delay_ > 0){
//								std::this_thread::sleep_for(std::chrono::duration<double>(option->rank0_delay_));
//							}
//							if(rc != MPI_SUCCESS){
//								std::cerr << "SendTask MPI Error" << std::endl;
//								exit(64);
//							}
//							cur_pos = send_message + sizeof(int) + sizeof(int);
//							cur_num = 0;
//							break;
//						}
//					}

				}

			} else{    //fail in send_queue.try_pop(p_col)
				std::this_thread::yield();
			}

		} // elapsed_seconds > timeout

		// send remaining columns to random machine
		if(cur_num != 0){
			int target_rank = target_dist(send_rng);
			_send_msg(send_message, cur_num, target_rank);

			local_send_count += cur_num;
		}

		// send dying message to every machine
		*(reinterpret_cast<int*>(send_message) + 1) = -(rank + 1);
		for(int i = 0; i < numtasks; i++){
			//tbb::tick_count t = tbb::tick_count::now();
			int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, i, MsgType::DATA, MPI_COMM_WORLD);
			if(rc != MPI_SUCCESS){
				std::cerr << "SendTask MPI Error" << std::endl;
				exit(64);
			}
			//do_net_control_ratio(msg_bytenum, tbb::tick_count::now() - t);
		}

		sallocator<char>().deallocate(send_message, msg_bytenum);

		cout << "send thread finishing" << endl;
	} //end of train_send_func

	/////////////////////////////////////////////////////////
	// Define Training Receive Function
	/////////////////////////////////////////////////////////
	void train_recv_func(){
		const int dim = option->latent_dimension_;
		char* recv_message = sallocator<char>().allocate(msg_bytenum);

		//const tick_count start_time = tick_count::now();
		//int monitor_num = 0;

		int num_dead = 0;

		MPI_Status status;

		while(num_dead < numtasks){

			//double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
			//if(monitor_num < elapsed_seconds){
			//	cout << "receiver thread alive," << rank << "," << monitor_num << endl;
			//	monitor_num++;
			//}

			int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//do_net_control_delay();

			if(rc != MPI_SUCCESS){
				std::cerr << "ReceiveTask MPI Error" << std::endl;
				exit(64);
			}

			if(status.MPI_TAG == MsgType::DATA){
				int queue_size = *(reinterpret_cast<int*>(recv_message));
				int num_received = *(reinterpret_cast<int*>(recv_message) + 1);
				//queue_current_sizes[status.MPI_SOURCE] = queue_size;

				// negative numbers are dying messages
				if(num_received < 0){
					num_dead++;
				} else{
					char* cur_pos = recv_message + sizeof(int) + sizeof(int);
					int source = status.MPI_SOURCE * option->num_threads_;
					for(int i = 0; i < num_received; i++){
						ColumnData* p_col = column_pool->pop();
						p_col->deserialize(cur_pos, dim);
						p_col->source_ = source;

						// generate permutation
						p_col->set_perm(option->num_threads_, rng);

						job_queues[p_col->perm_[p_col->pos_]].push(p_col);
						cur_pos += unit_bytenum;
					}
				}
			} else if(status.MPI_TAG == MsgType::CP_START){
				//push cp-start-signal to all threads
				for(int i = 0; i < option->num_threads_; ++i){
					ColumnData* p_col = column_pool->pop();
					p_col->col_index_ = cp_signal_start;
					//					p_col->source_=status.MPI_SOURCE;
					p_col->pos_ = *reinterpret_cast<int*>(recv_message);//epoch
					job_queues[i].push(p_col);
				}
			} else if(status.MPI_TAG == MsgType::CP_FLUSH){
				//push cp-flush-signal to all threads
				for(int i = 0; i < option->num_threads_; ++i){
					ColumnData* p_col = column_pool->pop();
					p_col->col_index_ = cp_signal_flush;
					p_col->pos_ = *reinterpret_cast<int*>(recv_message);//source
//					p_col->source_=status.MPI_SOURCE;
					job_queues[i].push(p_col);
				}
			}
		}
		sallocator<char>().deallocate(recv_message, msg_bytenum);
	} // end receiving for train

	/////////////////////////////////////////////////////////
	// Define Testing Sender Thread
	/////////////////////////////////////////////////////////
	void test_send_func(){
		const long mask = (1L << rank);

		const int dim = option->latent_dimension_;

		//		const tick_count start_time = tick_count::now();
		//		int monitor_num = 0;

		char* send_message = sallocator<char>().allocate(msg_bytenum);
		char* cur_pos = send_message + sizeof(int);
		int cur_num = 0;

		int send_count = 0;

		int target_rank = rank + 1;
		target_rank %= numtasks;

		while(send_count < global_num_cols){

			//			double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
			//			if(monitor_num < elapsed_seconds){
			//				cout << "test sender thread alive: " << monitor_num << endl;
			//				monitor_num++;
			//			}

			ColumnData* p_col;

			if(send_queue.try_pop(p_col)){

				// if the column was not already processed
				if((p_col->flag_ & mask) == 0){

					p_col->flag_ |= mask;

					//					*(reinterpret_cast<int *>(cur_pos)) = p_col->col_index_;
					//					*(reinterpret_cast<long *>(cur_pos + sizeof(int))) = p_col->flag_;
					//					scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(long) + sizeof(int));
					//					std::copy(p_col->values_, p_col->values_ + dim, dest);
					p_col->serialize(cur_pos, dim);

					cur_pos += unit_bytenum;
					cur_num++;

					send_count++;

					if(cur_num >= UNITS_PER_MSG){

						*(reinterpret_cast<int*>(send_message)) = cur_num;

						// choose destination
						int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);

						if(rc != MPI_SUCCESS){
							std::cerr << "SendTask MPI Error" << std::endl;
							exit(64);
						}

						cur_pos = send_message + sizeof(int);
						cur_num = 0;

					}

				} else{
					cout << "!!! should not happen! flag:" << p_col->flag_ << "???" << endl;
				}

				column_pool->push(p_col);

			} else{

				// even if pop was failed, if there is remaining message send it to another machine
				if(cur_num > 0){

					*(reinterpret_cast<int*>(send_message)) = cur_num;

					// choose destination
					int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);

					if(rc != MPI_SUCCESS){
						std::cerr << "SendTask MPI Error" << std::endl;
						exit(64);
					}

					cur_pos = send_message + sizeof(int);
					cur_num = 0;

				} else{
					std::this_thread::yield();
				}
			}

		}

		if(cur_num > 0){
			// send remaining columns to designated machine
			*(reinterpret_cast<int*>(send_message)) = cur_num;
			int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);

			if(rc != MPI_SUCCESS){
				std::cerr << "SendTask MPI Error" << std::endl;
				exit(64);
			}
		}

		sallocator<char>().deallocate(send_message, msg_bytenum);

		cout << "test send thread finishing," << rank << endl;

	} // end of test_send_func

	/////////////////////////////////////////////////////////
	// Define Testing Receive Function
	/////////////////////////////////////////////////////////
	void test_recv_func(){
		const int dim = option->latent_dimension_;
		char* recv_message = sallocator<char>().allocate(msg_bytenum);

		//		const tick_count start_time = tick_count::now();
		//		int monitor_num = 0;

		int recv_count = 0;

		MPI_Status status;

		const long mask = (1L << rank);

		while(recv_count < global_num_cols){

			//			double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
			//			if(monitor_num < elapsed_seconds){
			//				cout << "receiver thread alive: "<< monitor_num << endl;
			//				monitor_num++;
			//			}

			int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if(rc != MPI_SUCCESS){
				std::cerr << "ReceiveTask MPI Error" << std::endl;
				exit(64);
			}
			if(status.MPI_TAG != MsgType::DATA){
				cout << "received a signal while testing. signal " << status.MPI_TAG << ", from " << status.MPI_SOURCE << endl;
				continue;
			}

			int num_received = *(reinterpret_cast<int*>(recv_message));
			// negative numbers are dying messages
			char* cur_pos = recv_message + sizeof(int);
			for(int i = 0; i < num_received; i++){

				ColumnData* p_col = column_pool->pop();
				p_col->deserialize(cur_pos, dim);
				//				p_col->col_index_ = *(reinterpret_cast<int *>(cur_pos));
				//				p_col->flag_ = *(reinterpret_cast<long *>(cur_pos + sizeof(int)));
				//				scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(int)+ sizeof(long));
				//				std::copy(dest, dest + dim, p_col->values_);

				if((mask & p_col->flag_) == 0){
					job_queues[0].push(p_col);
				} else{
					// discard the column
					saved_columns.push_back(p_col);
				}

				cur_pos += unit_bytenum;
				recv_count++;
			}
		}
		sallocator<char>().deallocate(recv_message, msg_bytenum);
	}

	/////////////////////////////////////////////////////////
	// Define Checkpoint functions
	/////////////////////////////////////////////////////////
	void signal_handler_start(int thread_index, ColumnData* p_col, scalar* latent_rows, int local_num_rows, int dim){
		int epoch = p_col->pos_;
		int part_index = rank * option->num_threads_ + thread_index;
		//		cerr<<"starting cp "<<epoch<<" at "<<part_index<<endl;
		if(start_cp(thread_index, epoch)){
			cout << "checkpoint start: " << cp_epoch[thread_index] << " on " << rank << "," << thread_index << endl;
		} else{
			return;
		}
		//archive state
		archive_local(thread_index, latent_rows, local_num_rows, dim);
		//send flush signal part 1 (same MPI instance):
		for(int i = 0; i < option->num_threads_; ++i){
			ColumnData* p = column_pool->pop();
			p->col_index_ = cp_signal_flush;
			p->pos_ = part_index;
			job_queues[i].push(p);
		}
		//send flush signal part 2 (other MPI instance):
		p_col->col_index_ = cp_signal_flush; //set message type
		p_col->pos_ = part_index; // set source
		send_queue.push(p_col);
		//		cerr<<"send flush signal from "<<part_index<<endl;
		//		cerr<<thread_index<<" queue-len : "<<job_queues[thread_index].unsafe_size()<<endl;
	}
	void signal_handler_flush(int thread_index, ColumnData* p_col){
		//		int part_index = rank * option->num_threads_ + thread_index;
		int source = p_col->pos_;
		//		cerr<<"receive flush signal at "<<part_index<<" from "<<source<<endl;
		received_flush[thread_index][source] = true;
		++count_recv_flush[thread_index];
		if(count_recv_flush[thread_index] == num_parts){
			checkpointing[thread_index] = false;
			finish_cp(thread_index);
			cout << "checkpoint finish: " << cp_epoch[thread_index] << " on " << rank << "," << thread_index << endl;
		}
	}

	bool start_cp(int thread_index, int epoch){
		int part_index = rank * option->num_threads_ + thread_index;
		//		cerr<<"in-start-cp "<<thread_index<<endl;
		if(checkpointing[thread_index]){
			cerr << "on part " << part_index << " last checkpoint not finish, abandon new request: "
				<< cp_epoch[thread_index] << " vs " << epoch << endl;
			return false;
		} else if(cp_epoch[thread_index] >= epoch){
			cerr << "received checkpoint epoch (" << epoch << ") is not larger than last epoch (" << cp_epoch[thread_index] << ")" << endl;
			return false;
		}
		//init value
		cp_epoch[thread_index] = epoch;
		checkpointing[thread_index] = true;
		std::fill(received_flush[thread_index].begin(), received_flush[thread_index].end(), false);
		count_recv_flush[thread_index] = 0;
		//msg_archived[thread_index]=0;
		//create cp-files
		cp_folder = option->cp_path_ + (option->cp_path_.empty() ? "" : "/") +
			std::to_string(option->job_id_) + "/epoch-" + std::to_string(cp_epoch[thread_index]) + "/";
		//cerr<<cp_folder<<endl;
		boost::filesystem::path p(cp_folder);
		boost::filesystem::create_directories(p);

		part_index = rank * option->num_threads_ + thread_index;
		cp_fmsgs[thread_index] = new ofstream(cp_folder + std::to_string(part_index) + ".msg", ofstream::binary);
		return true;
	}
	void archive_local(int thread_index, scalar* latent_rows, int local_num_rows, int dim){
		int part_index = rank * option->num_threads_ + thread_index;
		tick_count t0 = tbb::tick_count::now();
		ofstream fout(cp_folder + std::to_string(part_index) + ".state", ofstream::binary);
		fout.write(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
		fout.write(reinterpret_cast<char*>(&dim), sizeof(dim));
		fout.write(reinterpret_cast<char*>(latent_rows), sizeof(scalar) * local_num_rows * dim);
		fout.close();
		cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
	}
	void archive_msg(int thread_index, ColumnData* p_col){
		char* buffer = new char[unit_bytenum];
		tick_count t0 = tbb::tick_count::now();
		p_col->serialize(buffer, option->latent_dimension_);
		cp_fmsgs[thread_index]->write(buffer, unit_bytenum);
		cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
		//++msg_archived[thread_index];
		delete[] buffer;
	}
	void finish_cp(int thread_index){
		cp_fmsgs[thread_index]->close();
		cp_fmsgs[thread_index]->clear();
		delete cp_fmsgs[thread_index];
		checkpointing[thread_index] = false;
		//cerr<<"Message archived at "<<cp_epoch[thread_index]<<" on "<<rank<<","<<thread_index
		//		<<" : "<<msg_archived[thread_index]<<" . queue-len : "<<job_queues[thread_index].unsafe_size()<<endl;
	}

	void restore_local(const string& cp_folder, int part_index, scalar* latent_rows, int& local_num_rows, int& dim){
		ifstream fin(cp_folder + std::to_string(part_index) + ".state", ofstream::binary);
		fin.read(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
		fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
		fin.read(reinterpret_cast<char*>(latent_rows), sizeof(scalar) * local_num_rows * dim);
		fin.close();
	}
	void restore_msg(const string& cp_folder, int part_index){
		char* buffer = new char[unit_bytenum];
		ifstream fin(cp_folder + std::to_string(part_index) + ".msg", ofstream::binary);
		while(fin){
			fin.read(buffer, unit_bytenum);
			ColumnData* p_col = column_pool->pop();
			p_col->deserialize(buffer, option->latent_dimension_);
			// Message is not restored to the previous thread. But it does not matter.
			p_col->set_perm(option->num_threads_, rng);
			job_queues[p_col->perm_[0]].push(p_col);
		}
		fin.close();
		delete[] buffer;
	}
	void restore(int epoch, int thread_index, scalar* latent_rows, int& local_num_rows, int& dim){
		cp_epoch[thread_index] = epoch;
		int part_index = rank * option->num_threads_ + thread_index;
		cp_folder = option->cp_path_ + (option->cp_path_.empty() ? "" : "/") +
			std::to_string(option->job_id_) + "/epoch-" + std::to_string(cp_epoch[thread_index]) + "/";
		restore_local(cp_folder, part_index, latent_rows, local_num_rows, dim);
		restore_msg(cp_folder, part_index);
	}

public:
	int run(int argc, char** argv){

		if(!initial(argc, argv)){
			return 1;
		}

		// count the number of threads in the machine which initial setup for training is done
		//atomic<int> count_setup_threads;
		count_setup_threads = 0;

		// this flag will be turned on when all threads are ready for training
		//atomic<bool> flag_train_ready;
		flag_train_ready = false;

		// this flag will be used to send signals to all threads that it has to stop training
		//atomic<bool> flag_train_stop;
		flag_train_stop = false;

		// this flag will be turned on when all threads are ready for testing
		//atomic<bool> flag_test_ready;
		flag_test_ready = false;

		// this flag will be used to send signals to all threads that it has to stop testing
		//atomic<bool> flag_test_stop;
		flag_test_stop = false;

		//std::mutex print_mutex;
		//std::condition_variable print_waiter;

		std::thread* master_thread = nullptr;
		if(rank == 0 && option->cp_interval_ > 0){
			master_thread = new std::thread(std::bind(&NomadBody::master_func, this));
		}

		wait_number = 0;

		// create and run updater threads
		std::thread* updater_threads = callocator<std::thread>().allocate(option->num_threads_);
		for(int i = 0; i < option->num_threads_; i++){
			callocator<std::thread>().construct(updater_threads + i, std::bind(&NomadBody::updater_func, this, i));
		}
		while(count_setup_threads < option->num_threads_){
			// wait until data loading and initializaiton of rows are done in every updater thread
			std::this_thread::yield();
		}

		/////////////////////////////////////////////////////////
		// Initialize Columns
		/////////////////////////////////////////////////////////

		rng_type rng(option->seed_ + rank * 131 + 139);
		std::uniform_real_distribution<scalar> init_dist(0, 1.0 / sqrt(option->latent_dimension_));

		int columns_per_machine = global_num_cols / numtasks + ((global_num_cols % numtasks > 0) ? 1 : 0);
		int col_start = columns_per_machine * rank;
		int col_end = std::min(columns_per_machine * (rank + 1), global_num_cols);

		// generate columns
		for(int i = col_start; i < col_end; i++){

			// create additional RNG, to make it identical to other programs
			rng_type rng_temp(option->seed_ + rank + 137);

			// create a column
			ColumnData* p_col = column_pool->pop();
			p_col->col_index_ = i;
			p_col->flag_ = 0;
			// create initial permutation for the column
//			p_col->pos_ = 0;
//			for(int j = 0; j < option->num_threads_; j++){
//				p_col->perm_[j] = j;
//			}
//			std::shuffle(p_col->perm_, p_col->perm_ + option->num_threads_, rng_temp);
			p_col->set_perm(option->num_threads_, rng_temp);

			// initialize parameter
			for(int j = 0; j < option->latent_dimension_; j++){
				p_col->values_[j] = init_dist(rng);
			}

			// push to the job queue
			job_queues[p_col->perm_[p_col->pos_]].push(p_col);
		}

		for(double ttt : option->timeouts_){
			cout << "timeout: " << ttt << endl;
		}

		// XXX: main working loop
		for(unsigned int main_timeout_iter = 0; main_timeout_iter < option->timeouts_.size(); main_timeout_iter++){

			const double timeout = (main_timeout_iter == 0) ? option->timeouts_[0] :
				option->timeouts_[main_timeout_iter] - option->timeouts_[main_timeout_iter - 1];

			// send thread for testing
			std::thread train_send_thread(std::bind(&NomadBody::train_send_func, this, timeout));

			// wait until every machine is ready
			MPI_Barrier(MPI_COMM_WORLD);

			/////////////////////////////////////////////////////////
			// Start Training
			/////////////////////////////////////////////////////////

			// now we are ready to train
			flag_train_ready = true;

			train_recv_func();

			train_send_thread.join();

			flag_train_stop = true;
			flag_train_ready = false;
			count_setup_threads = 0;

			// prepare for test
			{
				// gather everything that is within the machine
				vector<ColumnData*, sallocator<ColumnData*> > local_columns;

				int num_columns_prepared = 0;
				int global_num_columns_prepared = 0;

				while(global_num_columns_prepared < global_num_cols){

					for(int i = 0; i < option->num_threads_; i++){
						ColumnData* p_col;
						while(job_queues[i].try_pop(p_col)){
							local_columns.push_back(p_col);
							num_columns_prepared++;
						}
					}

					{
						ColumnData* p_col;
						while(send_queue.try_pop(p_col)){
							local_columns.push_back(p_col);
							num_columns_prepared++;
						}
					}

					MPI_Allreduce(&num_columns_prepared, &global_num_columns_prepared, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

					if(rank == 0){
						cout << "num columns prepared: " << global_num_columns_prepared << " / " << global_num_cols << endl;
						std::this_thread::sleep_for(std::chrono::duration<double>(0.2));
					}

				}

				for(ColumnData* p_col : local_columns){
					p_col->flag_ = 0;
					job_queues[0].push(p_col);
				}

			}

			// wait until every machine is ready
			MPI_Barrier(MPI_COMM_WORLD);

			/////////////////////////////////////////////////////////
			// Start Testing
			/////////////////////////////////////////////////////////

			// now start actual computation
			flag_test_ready = true;

			// send thread for testing
			std::thread test_send_thread(std::bind(&NomadBody::test_send_func, this));

			// receive columns for testing
			test_recv_func();
			cout << "test receive done," << rank << endl;

			test_send_thread.join();

			// test done
			flag_test_stop = true;

			cout << "waiting to join with updaters," << rank << endl;

			while(count_setup_threads < option->num_threads_){
				std::this_thread::yield();
			}

			/////////////////////////////////////////////////////////
			// Compute Statistics
			/////////////////////////////////////////////////////////

			long long machine_num_updates = 0; // std::accumulate(num_updates, num_updates + option->num_threads_, 0);
			for(int i = 0; i < option->num_threads_; i++){
				machine_num_updates += num_updates[i];
			}
			cout << "machine_num_updates: " << machine_num_updates << endl;

			long long machine_num_failures = 0; // std::accumulate(num_updates, num_updates + option->num_threads_, 0);
			for(int i = 0; i < option->num_threads_; i++){
				machine_num_failures += num_failures[i];
			}
			cout << "machine_num_failures: " << machine_num_failures << endl;

			int machine_train_count_error = std::accumulate(train_count_errors, train_count_errors + option->num_threads_, 0);
			int machine_test_count_error = std::accumulate(test_count_errors, test_count_errors + option->num_threads_, 0);
			real machine_train_sum_error = std::accumulate(train_sum_errors, train_sum_errors + option->num_threads_, 0.0);
			real machine_test_sum_error = std::accumulate(test_sum_errors, test_sum_errors + option->num_threads_, 0.0);

			int global_train_count_error = 0;
			MPI_Allreduce(&machine_train_count_error, &global_train_count_error, 1, MPI_INT,
				MPI_SUM, MPI_COMM_WORLD);
			int global_test_count_error = 0;
			MPI_Allreduce(&machine_test_count_error, &global_test_count_error, 1, MPI_INT,
				MPI_SUM, MPI_COMM_WORLD);

			real global_train_sum_error = 0.0;
			MPI_Allreduce(&machine_train_sum_error, &global_train_sum_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			real global_test_sum_error = 0.0;
			MPI_Allreduce(&machine_test_sum_error, &global_test_sum_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			long long global_num_updates = 0;
			MPI_Allreduce(&machine_num_updates, &global_num_updates, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

			long long global_num_failures = 0;
			MPI_Allreduce(&machine_num_failures, &global_num_failures, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

			long long global_send_count = 0;
			MPI_Allreduce(&local_send_count, &global_send_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

			int machine_col_empty = 0;
			for(int i = 0; i < global_num_cols; i++){
				if(is_column_empty[i]){
					machine_col_empty++;
				}
			}

			int global_col_empty = 0;
			MPI_Allreduce(&machine_col_empty, &global_col_empty, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			MPI_Barrier(MPI_COMM_WORLD);

			if(rank == 0){
				cout << "=====================================================" << endl;
				cout << "elapsed time: " << option->timeouts_[main_timeout_iter] << endl;
				cout << "current training RMSE: " << std::fixed << std::setprecision(10)
					<< sqrt(global_train_sum_error / global_train_count_error) << endl;
				cout << "current test RMSE: " << std::fixed << std::setprecision(10)
					<< sqrt(global_test_sum_error / global_test_count_error) << endl;

				cout << "testgrep," << numtasks << "," << option->num_threads_ << ","
					<< option->timeouts_[main_timeout_iter] << "," << global_num_updates << ","
					<< sqrt(global_test_sum_error / global_test_count_error)
					<< "," << global_test_sum_error << "," << global_test_count_error
					<< "," << global_num_failures << "," << global_col_empty
					<< "," << global_send_count << ","
					<< sqrt(global_train_sum_error / global_train_count_error)
					<< "," << global_train_sum_error << "," << global_train_count_error
					<< endl;
				cout << "=====================================================" << endl;
			}
			if(option->flag_pause_){
				std::this_thread::sleep_for(std::chrono::duration<double>(3.0));
			}

			// initialize state variables
			flag_train_ready = false;
			flag_train_stop = false;
			flag_test_ready = false;
			flag_test_stop = false;

			// BUGBUG: saved_columns: do initialization and push to job queue again
			for(ColumnData* p_col : saved_columns){

				p_col->flag_ = 0;
				// create initial permutation for the column
//				p_col->pos_ = 0;
//				for(int j = 0; j < option->num_threads_; j++){
//					p_col->perm_[j] = j;
//				}
//				std::shuffle(p_col->perm_, p_col->perm_ + option->num_threads_, rng);
				p_col->set_perm(option->num_threads_, rng);

				// push to the job queue
				job_queues[p_col->perm_[p_col->pos_]].push(p_col);

			}

			// if at the last iteration, do not clear this thing to print out to file
			if(main_timeout_iter < option->timeouts_.size() - 1){
				saved_columns.clear();
			}

		}  // end of timeout loop

		finished = true;
		cp_cv.notify_all();
		if(master_thread && master_thread->joinable()){
			master_thread->join();
			delete master_thread;
		}

		cout << "Waiting for updater threads to join" << endl;
		for(int i = 0; i < option->num_threads_; i++){
			updater_threads[i].join();
		}

		double machine_cp_write_time = std::accumulate(cp_write_time.begin(), cp_write_time.end(), 0.0);
		double global_cp_write_time = 0.0;
		MPI_Allreduce(&machine_cp_write_time, &global_cp_write_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if(rank == 0){
			cout << "Total write time for " << cp_master_epoch << " checkpoints " << global_cp_write_time
				<< " . Each one is " << global_cp_write_time / cp_master_epoch << endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if(option->output_path_.length() > 0){
			for(int task_iter = 0; task_iter < numtasks; task_iter++){
				if(task_iter == rank){
					ofstream ofs(option->output_path_ + std::to_string(rank), ofstream::out | ofstream::app);
					for(ColumnData* p_col : saved_columns){
						ofs << "column," << (p_col->col_index_);
						for(int t = 0; t < option->latent_dimension_; t++){
							ofs << "," << p_col->values_[t];
						}
						ofs << endl;
					}
					ofs.close();
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}

		cout << "All done, now free memory" << endl;

		callocator<colque>().deallocate(job_queues, option->num_threads_);

		for(int i = 0; i < option->num_threads_; i++){
			callocator<std::thread>().destroy(updater_threads + i);
			callocator<atomic<long long> >().destroy(num_updates + i);
			callocator<atomic<long long> >().destroy(num_failures + i);
		}
		callocator<atomic<long long> >().deallocate(num_updates, option->num_threads_);
		callocator<atomic<long long> >().deallocate(num_failures, option->num_threads_);

		callocator<std::thread>().deallocate(updater_threads, option->num_threads_);

		callocator<int>().deallocate(train_count_errors, option->num_threads_);
		callocator<real>().deallocate(train_sum_errors, option->num_threads_);
		callocator<int>().deallocate(test_count_errors, option->num_threads_);
		callocator<real>().deallocate(test_sum_errors, option->num_threads_);

		callocator<atomic<bool> >().deallocate(is_column_empty, global_num_cols);

		callocator<atomic<int> >().deallocate(queue_current_sizes, numtasks);

		delete column_pool;

		MPI_Finalize();

		return 0;

	}

};

#endif
