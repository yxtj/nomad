#include "nomad_body.h"

#include <tbb/tbb.h>
//#define TBB_IMPLEMENT_CPP0X
//#include <tbb/compat/thread>

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/format.hpp>

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
}

void NomadBody::initial_data4machine(){
	// used to compute the number of empty columns inside a machine
	// BUGBUG: right now it does not have any purpose other than computing statistics
	// we may enhance the efficiency of communication by taking advantage of this information
	is_column_empty = callocator<atomic<bool> >().allocate(global_num_cols);
	for(int i = 0; i < global_num_cols; i++){
		is_column_empty[i] = true;
	}
}

void NomadBody::initial_net_data(){
	// array used to remember the sizes of send_queue in each machine
	queue_current_sizes = callocator<atomic<int> >().allocate(numtasks);
	for(int i = 0; i < numtasks; i++){
		queue_current_sizes[i] = 0;
	}
	// we try to bound the size of send_queue's by this number
	queue_upperbound = global_num_cols * 4 / numtasks;

	// define constants needed for network communication
	// col_index + vector
	unit_bytenum = sizeof(int) + sizeof(long) + sizeof(double) * option->latent_dimension_;
	// current queue size + number of columns + columns
	msg_bytenum = sizeof(int) + sizeof(int) + unit_bytenum * UNITS_PER_MSG;
}

void NomadBody::initial_cp(){
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
	log_header = "W" + to_string(rank) + ": ";
	cout << log_header << boost::format("processor name: %s, number of tasks: %d, rank: %d") % hostname % numtasks % rank << endl;

	num_parts = numtasks * option->num_threads_;

	cout << log_header << "number of threads: " << option->num_threads_ << ", number of parts: " << num_parts << endl;

	// read number of columns
	global_num_cols = get_num_cols(option->path_);

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
	rng = mt19937_64(option->seed_ + rank * 131 + 139);

	// these arrays will be used to calculate test error
	// each thread will calculate test error on its own, and the results will be aggregated
	train_count_errors = callocator<int>().allocate(option->num_threads_);
	train_sum_errors = callocator<double>().allocate(option->num_threads_);
	test_count_errors = callocator<int>().allocate(option->num_threads_);
	test_sum_errors = callocator<double>().allocate(option->num_threads_);

	initial_cp();
	initial_net_control();
	finished = false;
	return true;
}