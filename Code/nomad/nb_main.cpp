#include "nomad_body.h"

#include <tbb/tbb.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <boost/format.hpp>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

#include "CheckpointState.h"

using namespace std;

//using tbb::atomic;
using tbb::tick_count;

using nomad::ColumnData;
using nomad::MsgType;


/////////////////////////////////////////////////////////
// Define message sending function
/////////////////////////////////////////////////////////
void NomadBody::_send_msg(char* send_message, const int cur_num, const int target_rank){
	*(reinterpret_cast<int*>(send_message)) = static_cast<int>(send_queue.unsafe_size());
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


/////////////////////////////////////////////////////////
// Define main running function
/////////////////////////////////////////////////////////
int NomadBody::run(NomadOption* opt){

	if(!initial(opt)){
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
	if(rank == 0 && option->cp_type_ != "none"){
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

	mt19937_64 rng(option->seed_ + rank * 131 + 139);
	std::uniform_real_distribution<double> init_dist(0, 1.0 / sqrt(option->latent_dimension_));

	int columns_per_machine = global_num_cols / numtasks + ((global_num_cols % numtasks > 0) ? 1 : 0);
	int col_start = columns_per_machine * rank;
	int col_end = std::min(columns_per_machine * (rank + 1), global_num_cols);

	// generate columns
	for(int i = col_start; i < col_end; i++){

		// create additional RNG, to make it identical to other programs
		mt19937_64 rng_temp(option->seed_ + rank + 137);

		// create a column
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = i;
		p_col->flag_ = 0;
		// create initial permutation for the column
		p_col->set_perm(option->num_threads_, rng_temp);

		// initialize parameter
		for(int j = 0; j < option->latent_dimension_; j++){
			p_col->values_[j] = init_dist(rng);
		}

		// push to the job queue
		job_queues[p_col->perm_[p_col->pos_]].push(p_col);
	}

	if(rank == 0){
		for(double ttt : option->timeouts_){
			cout << log_header << "timeout: " << ttt << endl;
		}
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
					cout << log_header << "num columns prepared: " << global_num_columns_prepared << " / " << global_num_cols << endl;
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
		cout << log_header << "test receive done," << rank << endl;

		test_send_thread.join();

		// test done
		flag_test_stop = true;

		cout << log_header << "waiting to join with updaters," << rank << endl;

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
		cout << log_header << "machine_num_updates: " << machine_num_updates << endl;

		long long machine_num_failures = 0; // std::accumulate(num_updates, num_updates + option->num_threads_, 0);
		for(int i = 0; i < option->num_threads_; i++){
			machine_num_failures += num_failures[i];
		}
		cout << log_header << "machine_num_failures: " << machine_num_failures << endl;

		int machine_train_count_error = std::accumulate(train_count_errors, train_count_errors + option->num_threads_, 0);
		int machine_test_count_error = std::accumulate(test_count_errors, test_count_errors + option->num_threads_, 0);
		double machine_train_sum_error = std::accumulate(train_sum_errors, train_sum_errors + option->num_threads_, 0.0);
		double machine_test_sum_error = std::accumulate(test_sum_errors, test_sum_errors + option->num_threads_, 0.0);

		int global_train_count_error = 0;
		MPI_Allreduce(&machine_train_count_error, &global_train_count_error, 1, MPI_INT,
			MPI_SUM, MPI_COMM_WORLD);
		int global_test_count_error = 0;
		MPI_Allreduce(&machine_test_count_error, &global_test_count_error, 1, MPI_INT,
			MPI_SUM, MPI_COMM_WORLD);

		double global_train_sum_error = 0.0;
		MPI_Allreduce(&machine_train_sum_error, &global_train_sum_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		double global_test_sum_error = 0.0;
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
			cout << log_header << "=====================================================" << endl;
			cout << log_header << "elapsed time: " << option->timeouts_[main_timeout_iter] << endl;
			cout << log_header << "current training RMSE: " << std::fixed << std::setprecision(10)
				<< sqrt(global_train_sum_error / global_train_count_error) << endl;
			cout << log_header << "current test RMSE: " << std::fixed << std::setprecision(10)
				<< sqrt(global_test_sum_error / global_test_count_error) << endl;

			cout << log_header << "testgrep," << numtasks << "," << option->num_threads_ << ","
				<< option->timeouts_[main_timeout_iter] << "," << global_num_updates << ","
				<< sqrt(global_test_sum_error / global_test_count_error)
				<< "," << global_test_sum_error << "," << global_test_count_error
				<< "," << global_num_failures << "," << global_col_empty
				<< "," << global_send_count << ","
				<< sqrt(global_train_sum_error / global_train_count_error)
				<< "," << global_train_sum_error << "," << global_train_count_error
				<< endl;
			cout << log_header << "=====================================================" << endl;
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

	cout << log_header << "Waiting for updater threads to join" << endl;
	for(int i = 0; i < option->num_threads_; i++){
		updater_threads[i].join();
	}

	if(rank == 0 && option->cp_type_ != "none"){
		double machine_cp_write_time = std::accumulate(cp_write_time.begin(), cp_write_time.end(), 0.0);
		double global_cp_write_time = 0.0;
		MPI_Allreduce(&machine_cp_write_time, &global_cp_write_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		cout << log_header << "Total write time for " << cp_master_epoch << " checkpoints " << global_cp_write_time
			<< " . Each one is " << global_cp_write_time / cp_master_epoch << endl;
	}

	// output column part
	if(option->output_path_.length() > 0){
		MPI_Barrier(MPI_COMM_WORLD);
		for(int task_iter = 0; task_iter < numtasks; task_iter++){
			if(task_iter == rank){
				ofstream ofs(option->output_path_ + std::to_string(rank), ofstream::out | ofstream::app);
				for(ColumnData* p_col : saved_columns){
					ofs << "column," << (p_col->col_index_);
					for(int t = 0; t < option->latent_dimension_; t++){
						ofs << "," << p_col->values_[t];
					}
					ofs << "\n";
				}
				ofs.close();
			}
			//MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	cout << log_header << "All done, now free memory" << endl;

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
	callocator<double>().deallocate(train_sum_errors, option->num_threads_);
	callocator<int>().deallocate(test_count_errors, option->num_threads_);
	callocator<double>().deallocate(test_sum_errors, option->num_threads_);

	callocator<atomic<bool> >().deallocate(is_column_empty, global_num_cols);

	callocator<atomic<int> >().deallocate(queue_current_sizes, numtasks);

	delete column_pool;

	MPI_Finalize();

	return 0;

}

/////////////////////////////////////////////////////////
// Define Updater Thread
/////////////////////////////////////////////////////////
void NomadBody::updater_func(int thread_index){
	int part_index = rank * option->num_threads_ + thread_index;
	string log_header = (boost::format("W%d-%d: ") % rank % thread_index).str();
	cout << log_header << boost::format("rank: %d, thread_index: %d, part_index: %d") % rank % thread_index % part_index << endl;

	/////////////////////////////////////////////////////////
	// Read Data
	/////////////////////////////////////////////////////////

	// each thread reads its own portion of data and stores in CSC format
	Data dtrain, dtest;

	bool succeed = load_train(option->path_, part_index, num_parts, thread_index == 0, dtrain);
	//min_row_index, local_num_rows, train_col_offset, train_row_idx, train_row_val);
	if(succeed == false){
		cerr << "error in reading training file" << endl;
		exit(11);
	}

	succeed = load_test(option->path_, part_index, num_parts, thread_index == 0, dtest);
	//min_row_index, local_num_rows, test_col_offset, test_row_idx, test_row_val);
	if(succeed == false){
		cerr << "error in reading testing file" << endl;
		exit(11);
	}

	vector<int, sallocator<int> >& train_col_offset = dtrain.col_offset, test_col_offset = dtest.col_offset;
	vector<int, sallocator<int> > train_row_idx = dtrain.row_idx, test_row_idx = dtest.row_idx;
	vector<double, sallocator<double> > train_row_val = dtrain.row_val, test_row_val = dtest.row_val;

	int local_num_rows = dtrain.local_num_rows;
	int min_row_index = dtrain.min_row_index;

	for(int i = 0; i < global_num_cols; i++){
		if(train_col_offset[i + 1] > train_col_offset[i]){
			is_column_empty[i] = true;
			//is_column_empty[i].compare_and_swap(false, true);
		}
	}

	/////////////////////////////////////////////////////////
	// Initialize Latent Data Structure
	/////////////////////////////////////////////////////////

	// now assign parameters for rows
	double* latent_rows = sallocator<double>().allocate(local_num_rows * option->latent_dimension_);

	// initialize random number generator
	mt19937_64 rng(option->seed_ + rank * 131 + thread_index + 1);
	std::uniform_real_distribution<double> init_dist(0, 1.0 / sqrt(option->latent_dimension_));
	for(int i = 0; i < local_num_rows * option->latent_dimension_; i++){
		latent_rows[i] = init_dist(rng);
	}

	int* col_update_counts = sallocator<int>().allocate(global_num_cols);
	std::fill_n(col_update_counts, global_num_cols, 0);

	// copy some essential parameters explicitly

	const int dim = option->latent_dimension_;
	const double learn_rate = option->learn_rate_;
	const double decay_rate = option->decay_rate_;
	const double par_lambda = option->par_lambda_;
	const int num_threads = option->num_threads_;
	const int num_reuse = option->num_reuse_;

	long long local_num_updates = 0;
	long long local_num_failures = 0;

	// notify that the thread is ready to run
	count_setup_threads++;

	for(unsigned int timeout_iter = 0; timeout_iter < option->timeouts_.size(); timeout_iter++){
		cout << log_header << "thread: " << thread_index << " ready to train!" << endl;
		// wait until all threads are ready
		while(flag_train_ready == false){
			std::this_thread::yield();
		}

		/////////////////////////////////////////////////////////
		// Training
		/////////////////////////////////////////////////////////

		while(flag_train_stop == false){
			if(cp_action_ready[thread_index]){
				cp_update_func_action(thread_index, part_index, latent_rows, local_num_rows);
				cp_action_ready[thread_index] = false;
			}

			if(!allow_processing || !allow_processing_thread[thread_index]){
				this_thread::yield();
				continue;
			}

			ColumnData* p_col;
			bool pop_succeed = job_queues[thread_index].try_pop(p_col);

			if(pop_succeed){
				// there was an available column in job queue to process
				// CP checking:
				if(p_col->col_index_ < 0){
					if(p_col->col_index_ == ColumnData::SIGNAL_CP_START){
						int epoch = p_col->pos_;
						cp_sht_start(thread_index, part_index, epoch, latent_rows, local_num_rows);
					} else if(p_col->col_index_ == ColumnData::SIGNAL_CP_CLEAR){
						int source = p_col->pos_;
						cp_sht_clear(thread_index, part_index, source, latent_rows, local_num_rows);
					} else if(p_col->col_index_ == ColumnData::SIGNAL_CP_RESUME){
						int epoch = p_col->pos_;
						cp_sht_resume(thread_index, part_index, epoch);
					}
					column_pool->push(p_col);
					continue;
				}

				// normal process:
				const int col_index = p_col->col_index_;
				const double step_size = learn_rate * 1.5
					/ (1.0 + decay_rate * pow(col_update_counts[col_index] + 1, 1.5));

				double* col = p_col->values_;

				// for each data point
				for(int offset = train_col_offset[col_index];
					offset < train_col_offset[col_index + 1]; offset++){

					// retrieve the point
					int row_index = train_row_idx[offset];
					double* row = latent_rows + dim * row_index;

					// calculate error
					double cur_error = std::inner_product(col, col + dim, row, -train_row_val[offset]);

					// update both row and column
					for(int i = 0; i < dim; i++){
						double tmp = row[i];

						row[i] -= step_size * (cur_error * col[i] + par_lambda * tmp);
						col[i] -= step_size * (cur_error * tmp + par_lambda * col[i]);
					}

					local_num_updates++;

				}

				col_update_counts[col_index]++;

				// CP: archive message (async)
				if(cp_need_archive_msg_from[thread_index][p_col->source_]){
					archive_msg(thread_index, p_col);
				}

				// send to the next thread
				p_col->pos_++;
				// if the column was circulated in every thread inside the machine, send to another machine
				if(p_col->pos_ >= num_threads * num_reuse){
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

		double train_sum_squared_error = 0.0;
		int train_count_error = 0;

		double test_sum_squared_error = 0.0;
		int test_count_error = 0;

		//int monitor_num = 0;
		//tbb::tick_count start_time = tbb::tick_count::now();

		// test until every column is processed
		while(num_col_processed < global_num_cols){

			//double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
			//if(monitor_num < elapsed_seconds){
			//	cout << log_header << "test updater alive," << rank << ","<< monitor_num << ","
			//			<< num_col_processed << "/" << global_num_cols << "" << endl;
			//	monitor_num++;
			//}

			ColumnData* p_col;

			if(job_queues[thread_index].try_pop(p_col)){

				double* col = p_col->values_;
				const int col_index = p_col->col_index_;

				// for each training data point
				for(int offset = train_col_offset[col_index];
					offset < train_col_offset[col_index + 1]; offset++){

					// retrieve the point
					int row_index = train_row_idx[offset];
					double* row = latent_rows + dim * row_index;

					// calculate error
					double cur_error = -train_row_val[offset];
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
					double* row = latent_rows + dim * row_index;

					// calculate error
					double cur_error = -test_row_val[offset];
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

	} // timeout list

	// print to the file
	if(option->output_path_.length() > 0){
		// output thread by thread
		while(wait_number < part_index % option->num_threads_){
			std::this_thread::yield();
		}

		ofstream::openmode mode = (part_index % option->num_threads_ == 0) ?
			ofstream::out : (ofstream::out | ofstream::app);
		ofstream ofs(option->output_path_ + std::to_string(rank), mode);

		cout << log_header << "min_row_index: " << min_row_index << endl;
		for(int i = 0; i < local_num_rows; i++){
			double* row = latent_rows + dim * i;
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

	sallocator<double>().deallocate(latent_rows, local_num_rows * option->latent_dimension_);
}
