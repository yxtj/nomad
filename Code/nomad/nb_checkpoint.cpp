#include "nomad_body.h"

#include <tbb/tbb.h>

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

using namespace std;

/////////////////////////////////////////////////////////
// Define Basic Checkpoint functions
/////////////////////////////////////////////////////////
void NomadBody::_send_clear_signal(bool send2self, bool direct_send)
{
	if(direct_send){
		int source = mpi_rank;
		for(int target_rank = 0; target_rank < mpi_size; ++target_rank){
			// including itself
			if(!send2self && target_rank == mpi_rank)
				continue;
			send_queue_force.emplace(ColumnData::SIGNAL_CP_CLEAR, target_rank);
		}
	} else{
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_CLEAR; //set message type
		p_col->pos_ = mpi_rank; // set source
		send_queue.push(p_col);
	}
}

void NomadBody::_send_lfinish_signal()
{
	send_queue_force.emplace(ColumnData::SIGNAL_CP_LFINISH, cp_epoch);
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
void NomadBody::archive_local(int thread_index, double* latent_rows, int local_num_rows){
	tick_count t0 = tbb::tick_count::now();
	int part_index = mpi_rank * option->num_threads_ + thread_index;
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
	int part_index = mpi_rank * option->num_threads_ + thread_index;
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
	int part_index = mpi_rank * option->num_threads_ + thread_index;
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

void NomadBody::_send_sig2threads_clear(int source)
{
	for(int i = 0; i < option->num_threads_; ++i){
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_CLEAR;
		//p_col->source_=status.MPI_SOURCE;
		p_col->pos_ = source;
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
	//cout << mpi_rank << " m start" << endl;
	cp_epoch = epoch;
	checkpointing = true;
	if(option->cp_type_ == "sync"){
		allow_sending = false;
		allow_processing = false;
		for(int i = 0; i < option->num_threads_; ++i){
			cp_action_ready[i] = true;
		}
	}else if(option->cp_type_ == "async"){
		_send_sig2threads_start(epoch);
		_send_clear_signal(true, false);
	}else if(option->cp_type_ == "vs"){
		allow_sending = false;
		_send_clear_signal(true, true);
	} else{

	}
}

void NomadBody::cp_shm_clear(int source)
{
	//cout << mpi_rank << " m clear " << source << endl;
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){
		_send_sig2threads_clear(source);
	} else if(option->cp_type_ == "vs"){
		++cp_received_clear_counter;
		if(cp_received_clear_counter == mpi_size){
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
	//cout << mpi_rank << " m resume" << endl;
	if(epoch != cp_epoch){
		cerr << "ERROR: epoch of checkpoint does not match: " << cp_epoch << " vs " << epoch << endl;
		exit(2);
	}
	if(option->cp_type_ == "sync"){
		archive_msg_queue_all();
		allow_processing = true;
		allow_sending = true;
	} else if(option->cp_type_ == "async"){
		for(int i = 0; i < option->num_threads_; ++i){
			cp_need_archive_msg_counter[i] = 0;
			for(int j = 0; j < mpi_size; ++j)
				cp_need_archive_msg_from[i][j] = false;
		}
	} else if(option->cp_type_ == "vs"){
		allow_sending = true;
		cp_received_clear_counter = 0;
	} else{

	}
	checkpointing = false;
	cp_ut_wait_counter = 0;
}

// thread level

void NomadBody::cp_sht_start(int thread_index, int part_index, int epoch, double* latent_rows, int local_num_rows)
{
	//cout << mpi_rank << "-" << thread_index << " t start" << endl;
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){
		for(int i = 0; i < mpi_size; ++i)
			cp_need_archive_msg_from[thread_index][i] = true;
		archive_local(thread_index, latent_rows, local_num_rows);
		cp_fmsgs[thread_index] = new ofstream(gen_cp_file_name(part_index) + ".msg", ofstream::binary);
	} else if(option->cp_type_ == "vs"){
		// nothing
	} else{

	}
}

void NomadBody::cp_sht_clear(int thread_index, int part_index, int source, double* latent_rows, int local_num_rows)
{
	//cout << mpi_rank << "-" << thread_index << " t clear: " << source << endl;
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){
		cp_need_archive_msg_from[thread_index][source] = false;
		++cp_need_archive_msg_counter[thread_index];
		if(cp_need_archive_msg_counter[thread_index] == mpi_size){
			cp_fmsgs[thread_index]->close();
			cp_fmsgs[thread_index]->clear();
			delete cp_fmsgs[thread_index];
			if(thread_index == 0){
				_send_lfinish_signal();
			}
		}
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
		// nothing
	} else if(option->cp_type_ == "vs"){
		// nothing
	} else{

	}
}

void NomadBody::cp_update_func_action(int thread_index, int part_index, double* latent_rows, int local_num_rows)
{
	//cout << mpi_rank << "-" << thread_index << " cp_uf " << endl;
	if(option->cp_type_ == "sync"){
		// triggered by start signal
		_sync_all_update_thread();
		archive_local(thread_index, latent_rows, local_num_rows);
		// message queues are not stable now
		if(thread_index == 0){
			_send_lfinish_signal();
		}
	} else if(option->cp_type_ == "async"){
		// nothing
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
