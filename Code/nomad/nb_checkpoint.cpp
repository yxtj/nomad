#include "nomad_body.h"

#include <tbb/tbb.h>

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <mpi.h>
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

using namespace std;

/////////////////////////////////////////////////////////
// Define Basic Checkpoint functions
/////////////////////////////////////////////////////////
void NomadBody::_send_clear_signal(bool send2self, bool direct_send)
{
	if(direct_send){
		int source = rank;
		for(int target_rank = 0; target_rank < numtasks; ++target_rank){
			if(target_rank == rank)
				continue;
			int rc = MPI_Ssend(reinterpret_cast<void*>(&source), sizeof(source), MPI_CHAR, target_rank, MsgType::CP_CLEAR, MPI_COMM_WORLD);
			if(rc != MPI_SUCCESS){
				std::cerr << "SendTask MPI Error" << std::endl;
				exit(64);
			}
		}
	} else{
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_CLEAR; //set message type
		p_col->pos_ = rank; // set source
		send_queue.push(p_col);
	}
}

void NomadBody::_send_lfinish_signal()
{
	int data[2];
	data[0] = rank;
	data[1] = cp_epoch;
	int rc = MPI_Ssend(reinterpret_cast<char*>(data), sizeof(data), MPI_CHAR, 0, MsgType::CP_LFINISH, MPI_COMM_WORLD);
	if(rc != MPI_SUCCESS){
		std::cerr << "SendTask MPI Error" << std::endl;
		exit(64);
	}
}

string NomadBody::gen_cp_file_name(int part_index)
{
	string res = option->cp_path_ + (option->cp_path_.empty() ? "" : "/") +
		(boost::format("cp-t%d-e%d-p%d") % option->job_id_ % cp_epoch % part_index).str();
	return res;
}

/////////////////////////////////////////////////////////
// Define Checkpoint Core Functions
/////////////////////////////////////////////////////////
bool NomadBody::start_cp(int thread_index, int epoch){
	int part_index = rank * option->num_threads_ + thread_index;
	//cerr<<"in-start-cp "<<thread_index<<endl;
	if(checkpointing[thread_index]){
		cerr << "on part " << part_index << " last checkpoint not finish, abandon new request: "
			<< cp_epoch << " vs " << epoch << endl;
		return false;
	} else if(cp_epoch >= epoch){
		cerr << "received checkpoint epoch (" << epoch << ") is not larger than last epoch (" << cp_epoch << ")" << endl;
		return false;
	}
	//init value
	cp_epoch = epoch;
	checkpointing[thread_index] = true;
	std::fill(received_flush[thread_index].begin(), received_flush[thread_index].end(), false);
	count_recv_flush[thread_index] = 0;
	//msg_archived[thread_index]=0;
	//create cp-files
	cp_folder = option->cp_path_ + (option->cp_path_.empty() ? "" : "/");// +
		//std::to_string(option->job_id_) + "/epoch-" + std::to_string(cp_epoch) + "/";
	//cerr<<cp_folder<<endl;
	boost::filesystem::path p(cp_folder);
	boost::filesystem::create_directories(p);
	
	if(option->cp_type_ == "async"){
		cp_fmsgs[thread_index] = new ofstream(gen_cp_file_name(part_index) + ".msg", ofstream::binary);
	}
	return true;
}

void NomadBody::archive_local(int thread_index, double* latent_rows, int local_num_rows){
	tick_count t0 = tbb::tick_count::now();
	int part_index = rank * option->num_threads_ + thread_index;
	int dim = option->latent_dimension_;
	ofstream fout(gen_cp_file_name(part_index) + ".state", ofstream::binary);
	fout.write(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
	fout.write(reinterpret_cast<char*>(&dim), sizeof(dim));
	fout.write(reinterpret_cast<char*>(latent_rows), sizeof(double) * local_num_rows * dim);
	fout.close();
	cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
}

void NomadBody::_archive_msg_queue(int thread_index, const string& suffix, colque& queue, bool locked)
{
	tick_count t0 = tbb::tick_count::now();
	int part_index = rank * option->num_threads_ + thread_index;
	char* buffer = new char[unit_bytenum];
	colque temp;
	ofstream fout(gen_cp_file_name(part_index) + suffix, ofstream::binary);
	if(locked){
		for(auto it = queue.unsafe_begin(); it != queue.unsafe_end(); ++it){
			(*it)->serialize(buffer, option->latent_dimension_);
			fout.write(buffer, unit_bytenum);
		}
		fout.close();
	}else{
		ColumnData* p_col = nullptr;
		while(queue.try_pop(p_col)){
			p_col->serialize(buffer, option->latent_dimension_);
			fout.write(buffer, unit_bytenum);
			temp.push(p_col);
		}
		fout.close();
		//queue = move(temp);
		while(temp.try_pop(p_col)){
			queue.push(p_col);
		}
	}
	cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
}

void NomadBody::arhive_job_queue(int thread_index, bool locked)
{
	_archive_msg_queue(thread_index, ".msg", job_queues[thread_index], locked);
}

void NomadBody::arhive_send_queue(bool locked)
{
	_archive_msg_queue(0, ".smsg", send_queue, locked);
}

void NomadBody::archive_msg_queue_all(bool locked)
{
	tick_count t0 = tbb::tick_count::now();
	// each job queue
	for(int thread_index = 0; thread_index < option->num_threads_; ++thread_index){
		arhive_job_queue(thread_index);
	}
	// send queue
	arhive_send_queue();
	cp_write_time[0] += (tbb::tick_count::now() - t0).seconds();

}

void NomadBody::archive_msg(int thread_index, ColumnData* p_col){
	char* buffer = new char[unit_bytenum];
	tick_count t0 = tbb::tick_count::now();
	p_col->serialize(buffer, option->latent_dimension_);
	cp_fmsgs[thread_index]->write(buffer, unit_bytenum);
	cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
	//++msg_archived[thread_index];
	delete[] buffer;
}

void NomadBody::finish_cp(int thread_index){
	if(option->cp_type_ == "async"){
		cp_fmsgs[thread_index]->close();
		cp_fmsgs[thread_index]->clear();
		delete cp_fmsgs[thread_index];
	}
	checkpointing[thread_index] = false;
	//cerr<<"Message archived at "<<cp_epoch[thread_index]<<" on "<<rank<<","<<thread_index
	//		<<" : "<<msg_archived[thread_index]<<" . queue-len : "<<job_queues[thread_index].unsafe_size()<<endl;
}

void NomadBody::restore_local(const string& cp_f, int part_index, double* latent_rows, int& local_num_rows, int& dim){
	ifstream fin(cp_f, ofstream::binary);
	fin.read(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
	fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
	fin.read(reinterpret_cast<char*>(latent_rows), sizeof(double) * local_num_rows * dim);
	fin.close();
}

void NomadBody::restore_msg_queue(const string& cp_f, colque& queue){
	char* buffer = new char[unit_bytenum];
	ifstream fin(cp_f, ofstream::binary);
	while(fin){
		fin.read(buffer, unit_bytenum);
		ColumnData* p_col = column_pool->pop();
		p_col->deserialize(buffer, option->latent_dimension_);
		queue.push(p_col);
	}
	fin.close();
	delete[] buffer;
}

void NomadBody::restore(int epoch, int thread_index, double* latent_rows, int& local_num_rows, int& dim){
	cp_epoch = epoch;
	int part_index = rank * option->num_threads_ + thread_index;
	string cp_f = gen_cp_file_name(part_index);
	restore_local(cp_f + ".state", part_index, latent_rows, local_num_rows, dim);
	for(int thread_index = 0; thread_index < option->num_threads_; ++thread_index){
		restore_msg_queue(cp_f + ".msg", job_queues[thread_index]);
	}
	restore_msg_queue(cp_f + ".smsg", send_queue);
}


void NomadBody::_send_sig2threads_start(int epoch)
{
	for(int i = 0; i < option->num_threads_; ++i){
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_START;
		//p_col->source_=status.MPI_SOURCE;
		p_col->pos_ = epoch;
		job_queues[i].push(p_col);
	}
}

void NomadBody::_send_sig2threads_clear(int source_part)
{
	for(int i = 0; i < option->num_threads_; ++i){
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_CLEAR;
		//p_col->source_=status.MPI_SOURCE;
		p_col->pos_ = source_part;
		job_queues[i].push(p_col);
	}
}

void NomadBody::_send_sig2threads_resume(int epoch)
{

	for(int i = 0; i < option->num_threads_; ++i){
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_RESUME;
		//p_col->source_=status.MPI_SOURCE;
		p_col->pos_ = epoch;
		job_queues[i].push(p_col);
	}
}

void NomadBody::_sync_all_update_thread()
{
	++cp_ut_wait_counter;
	while(cp_ut_wait_counter < option->num_threads_){
		this_thread::yield();
	}
}

// machine level

void NomadBody::cp_shm_start(int epoch)
{
	if(option->cp_type_ == "sync"){
		allow_sending = false;
		allow_processing = false;
		for(int i = 0; i < option->num_threads_; ++i){
			cp_action_ready[i] = true;
		}
	}else if(option->cp_type_ == "async"){
		_send_sig2threads_start(epoch);
	}else if(option->cp_type_ == "vs"){
		allow_sending = false;
		_send_clear_signal(true, true);
	} else{

	}
}

void NomadBody::cp_shm_clear(int source)
{
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){
		_send_sig2threads_clear(source);
	} else if(option->cp_type_ == "vs"){
		++cp_received_clear_counter;
		if(cp_received_clear_counter == numtasks){
			// temporarily stop processing to sablize the send_queue
			allow_processing = false;
			for(int i = 0; i < option->num_threads_; ++i)
				cp_action_ready[i] = true;
		}
	} else{

	}
}

void NomadBody::cp_shm_resume(int epoch)
{
	if(option->cp_type_ == "sync"){
		archive_msg_queue_all();
		allow_processing = true;
		allow_sending = true;
	} else if(option->cp_type_ == "async"){
		_send_sig2threads_resume(epoch);
	} else if(option->cp_type_ == "vs"){
		allow_sending = true;
		cp_received_clear_counter = 0;
	} else{

	}
	cp_ut_wait_counter = 0;
}

// thread level

void NomadBody::cp_sht_start(int thread_index, int part_index, int epoch, double* latent_rows, int local_num_rows)
{
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){

	} else if(option->cp_type_ == "vs"){
		_send_clear_signal(false, true);
	} else{
		// nothing
	}
}

void NomadBody::cp_sht_clear(int thread_index, int part_index, int source, double* latent_rows, int local_num_rows)
{
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){

	} else if(option->cp_type_ == "vs"){
		// nothing
	} else{

	}
}

void NomadBody::cp_sht_resume(int thread_index, int part_index, int epoch)
{
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){

	} else if(option->cp_type_ == "vs"){
		// nothing
	} else{

	}
}

void NomadBody::cp_update_func_action(int thread_index, int part_index, double* latent_rows, int local_num_rows)
{
	if(option->cp_type_ == "sync"){
		// triggered by start signal
		_sync_all_update_thread();
		archive_local(thread_index, latent_rows, local_num_rows);
		// message queues are not stable now
		if(thread_index == 0){
			_send_lfinish_signal();
		}
	} else if(option->cp_type_ == "async"){

	} else if(option->cp_type_ == "vs"){
		// triggered by the last clear singal
		_sync_all_update_thread();
		archive_local(thread_index, latent_rows, local_num_rows);
		arhive_job_queue(thread_index);
		// message queues are stable now
		if(thread_index == 0){
			arhive_send_queue();
			// resume processing
			allow_processing = true;
			_send_lfinish_signal();
		}
	} else{

	}

}

/////////////////////////////////////////////////////////
// Define Checkpoint functions for ASYNC
/////////////////////////////////////////////////////////
void NomadBody::signal_handler_start(int thread_index, ColumnData* p_col, double* latent_rows, int local_num_rows){
	int epoch = p_col->pos_;
	int part_index = rank * option->num_threads_ + thread_index;
	//cerr<<"starting cp "<<epoch<<" at "<<part_index<<endl;
	if(start_cp(thread_index, epoch)){
		cout << log_header << "checkpoint start: " << cp_epoch << " on " << rank << "," << thread_index << endl;
	} else{
		return;
	}
	// archive state
	archive_local(thread_index, latent_rows, local_num_rows);
	_send_clear_signal(true, false);
	//cerr<<"send flush signal from "<<part_index<<endl;
	//cerr<<thread_index<<" queue-len : "<<job_queues[thread_index].unsafe_size()<<endl;
}

void NomadBody::signal_handler_clear(int thread_index, ColumnData* p_col){
	//int part_index = rank * option->num_threads_ + thread_index;
	int source = p_col->pos_;
	//cerr<<"receive flush signal at "<<part_index<<" from "<<source<<endl;
	received_flush[thread_index][source] = true;
	++count_recv_flush[thread_index];
	if(count_recv_flush[thread_index] == num_parts){
		checkpointing[thread_index] = false;
		finish_cp(thread_index);
		cout << log_header << "checkpoint finish: " << cp_epoch << " on " << rank << "," << thread_index << endl;
	}
}

void NomadBody::signal_handler_resume(int thread_index, ColumnData* p_col)
{
	int epoch = p_col->pos_;
	if(epoch != cp_epoch){
		cerr << log_header << "checkpoint epoch does not match in resume: " << cp_epoch << " vs " << epoch;
		exit(5);
	}
	allow_sending = true;
}

/////////////////////////////////////////////////////////
// Define Checkpoint functions for SYNC
/////////////////////////////////////////////////////////
void NomadBody::cp_sht_start_sync(int thread_index, ColumnData* p_col, double* latent_rows, int local_num_rows)
{
	allow_sending = false;
	
}

void NomadBody::cp_sht_clear_sync(int thread_index, ColumnData* p_col)
{

}


