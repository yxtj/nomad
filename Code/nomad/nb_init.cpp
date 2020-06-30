#include "nomad_body.h"

#include <tbb/tbb.h>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/format.hpp>
#include <glog/logging.h>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32


using namespace std;


bool NomadBody::initial_mpi(){
	int mpi_thread_provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
	if(mpi_thread_provided != MPI_THREAD_MULTIPLE && mpi_thread_provided != MPI_THREAD_SERIALIZED){
		LOG(ERROR) << "MPI multiple thread not provided!!! (" << mpi_thread_provided << " != "
			<< MPI_THREAD_MULTIPLE << " or " << MPI_THREAD_SERIALIZED << ")" << endl;
		return false;
	}

	// retrieve MPI task info
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	int hostname_len;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(hostname, &hostname_len);
	hostname[hostname_len] = '\0';
	this->hostname = hostname;
	return true;
}

void NomadBody::initial_dataset()
{
	dstrain.resize(option->num_threads_);
	dstest.resize(option->num_threads_);
	local_thread_num_rows.assign(option->num_threads_, 0);
	local_thread_num_nonzero.assign(option->num_threads_, 0ll);
	for(int thread_index = 0; thread_index < option->num_threads_; ++thread_index){
		int part_index = mpi_rank * option->num_threads_ + thread_index;
		{
			bool succeed = load_train(option->path_, part_index, num_parts, dstrain[thread_index]);
			//min_row_index, local_num_rows, train_col_offset, train_row_idx, train_row_val);
			Data& d = dstrain[thread_index];
			LOG(INFO) << "Train data part " << part_index << ": "
				<< "nrows: " << d.num_rows << ", ncols: " << d.num_cols << ", total_nnz: " << d.num_nonzero << ", "
				<< "min_row: " << d.min_row_index << ", max_row: " << d.min_row_index + d.local_num_rows << ", "
				<< "nnz: " << d.local_num_nonzero << endl;
			LOG_IF(FATAL, !succeed) << "error in reading training file" << endl;
		}
		{
			bool succeed = load_test(option->path_, part_index, num_parts, dstest[thread_index]);
			//min_row_index, local_num_rows, test_col_offset, test_row_idx, test_row_val);
			Data& d = dstrain[thread_index];
			LOG(INFO) << "Test data part " << part_index << ": "
				<< "nrows: " << d.num_rows << ", ncols: " << d.num_cols << ", total_nnz: " << d.num_nonzero << ", "
				<< "min_row: " << d.min_row_index << ", max_row: " << d.min_row_index + d.local_num_rows << ", "
				<< "nnz: " << d.local_num_nonzero << endl;
			LOG_IF(FATAL, !succeed) << "error in reading test file" << endl;
		}
		global_num_cols = dstrain[thread_index].num_cols;
		global_num_rows = dstrain[thread_index].num_rows;
		global_num_nonzero = dstrain[thread_index].num_nonzero;

		local_thread_num_rows[thread_index] = dstrain[thread_index].local_num_rows;
		local_thread_num_nonzero[thread_index] = dstrain[thread_index].local_num_nonzero;
	}
}

void NomadBody::initial_data4thread(){
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

	allow_processing_thread = callocator<atomic<bool> >().allocate(option->num_threads_);
	for(int i = 0; i < option->num_threads_; i++){
		allow_processing_thread[i] = true;
	}
}

void NomadBody::initial_data4machine(){

}

void NomadBody::initial_net_data(){
	// array used to remember the sizes of send_queue in each machine
	queue_current_sizes = callocator<atomic<int> >().allocate(mpi_size);
	for(int i = 0; i < mpi_size; i++){
		queue_current_sizes[i] = 0;
	}
	// we try to bound the size of send_queue's by this number
	queue_upperbound = global_num_cols * 4 / mpi_size;

	column_per_msg = min(option->column_per_msg, global_num_cols / mpi_size / 4);
	column_per_msg = max(1, column_per_msg);

	// define constants needed for network communication
	// col_index + vector
	unit_bytenum = sizeof(int) + sizeof(long) + sizeof(double) * option->latent_dimension_;
	// current queue size + number of columns + columns
	msg_bytenum = msg_head_bytenum + unit_bytenum * column_per_msg;
}

void NomadBody::initial_termcheck()
{
	// master
	tm_min_updated_col = max(1, static_cast<int>(global_num_cols * option->termcheck_min_portion));
	if(mpi_rank == 0){
		tm_local_error_received.assign(mpi_size, 0.0);
		tm_local_error_ready = callocator<atomic<bool> >().allocate(mpi_size);
		for(int i = 0; i < mpi_size; ++i)
			tm_local_error_ready[i] = false;
		tm_local_update_count.assign(mpi_size, 0ll);
		tm_global_update_count = 0ll;
	}
	// worker
	tm_col_error.assign(global_num_cols, 0.0);
	tm_col_error_sum = 0.0;
}

void NomadBody::initial_cp(){
	//master:
	cp_master_epoch = 0;
	cp_master_lfinish_count = 0;
	//worker:
	cp_epoch = -1;
	cp_ut_wait_counter = 0;
	checkpointing = false;
	cp_received_clear.assign(mpi_size, false);
	cp_received_clear_counter = 0;
	//received_flush.resize(option->num_threads_, vector<bool>(num_parts, false));
	cp_need_archive_msg_from.resize(option->num_threads_);
	cp_need_archive_msg_counter = callocator<atomic<int>>().allocate(option->num_threads_);
	for(int i = 0; i < option->num_threads_; ++i){
		cp_need_archive_msg_counter[i] = 0;
		cp_need_archive_msg_from[i] = callocator<atomic<bool>>().allocate(mpi_size);
		for(int j = 0; j < mpi_size; ++j)
			cp_need_archive_msg_from[i][j] = false;
	}
	//count_recv_flush.resize(option->num_threads_, 0);
	cp_fmsgs.resize(option->num_threads_);
	//cp_state.resize(option->num_threads_, CheckpointState(num_parts));
	msg_archived.resize(option->num_threads_, 0);
	cp_time_write.resize(option->num_threads_, 0.0);
	cp_time_total_worker = 0.0;
	cp_time_total_master = 0.0;
	cp_action_ready = callocator<atomic<bool>>().allocate(option->num_threads_);
	for(int i = 0; i < option->num_threads_; ++i)
		cp_action_ready[i] = false;
}

void NomadBody::initial_net_control(){
	control_net_delay = option->net_delay > 0;
	if(control_net_delay){
		net_delay = std::chrono::duration<double>(option->net_delay);
	}
	control_net_ratio = option->net_delay != std::numeric_limits<double>::max();
	net_ratio = option->net_ratio;
}
bool NomadBody::initial(NomadOption* opt){
	option = opt;
	if(!initial_mpi())
		return false;
	log_header = "W" + to_string(mpi_rank) + ": ";
	LOG(INFO) << log_header << boost::format("processor name: %s, number of tasks: %d, rank: %d") % hostname % mpi_size % mpi_rank << endl;

	num_parts = mpi_size * option->num_threads_;

	LOG(INFO) << log_header << "number of threads: " << option->num_threads_ << ", number of parts: " << num_parts << endl;

	initial_dataset();
	// global_num_cols and global_num_nonzero are ready now.

	initial_data4thread();
	initial_data4machine();
	initial_net_data();

	// create a column pool with big enough size
	// this serves as a memory pool. global_num_cols * 3 / num_parts is arbitrary big enough number.
	// when the capacity is exceeded, it automatically assigns additional memory.
	// therefore no need to worry too much
	column_pool = new Pool(option->latent_dimension_, option->num_threads_);

	// for updater_func
	wait_number = 0;
	local_send_count = 0;

	// distribution used to initialize parameters
	// distribution is taken from Hsiang-Fu's implementation of DSGD
	rng = mt19937_64(option->seed_ + mpi_rank * 131 + 139);

	// these arrays will be used to calculate test error
	// each thread will calculate test error on its own, and the results will be aggregated
	train_count_errors = callocator<int>().allocate(option->num_threads_);
	train_sum_errors = callocator<double>().allocate(option->num_threads_);
	test_count_errors = callocator<int>().allocate(option->num_threads_);
	test_sum_errors = callocator<double>().allocate(option->num_threads_);

	initial_termcheck();
	initial_cp();
	initial_net_control();
	allow_sending = true;
	allow_processing = true;
	finished = false;
	return true;
}
