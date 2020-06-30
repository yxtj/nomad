#include "nomad_body.h"

#include <tbb/tbb.h>

#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

using namespace std;

/////////////////////////////////////////////////////////
// Define Basic Checkpoint functions
/////////////////////////////////////////////////////////
void NomadBody::_send_clear_signal(bool direct_send)
{
	if(direct_send){
		// including itself
		send_queue_force.emplace(ColumnData::SIGNAL_CP_CLEAR, cp_epoch);
	} else{
		ColumnData* p_col = column_pool->pop();
		p_col->col_index_ = ColumnData::SIGNAL_CP_CLEAR; //set message type
		p_col->pos_ = cp_epoch; // set epoch
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
// Define Checkpoint IO Functions
/////////////////////////////////////////////////////////
void NomadBody::archive_local(int thread_index, double* latent_rows, int local_num_rows)
{
	VLOG(2) << "W" << mpi_rank << ": archive local state";
	tbb::tick_count t0 = tbb::tick_count::now();
	int part_index = mpi_rank * option->num_threads_ + thread_index;
	int dim = option->latent_dimension_;
	ofstream fout(gen_cp_file_name(part_index) + ".state", ofstream::binary);
	fout.write(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
	fout.write(reinterpret_cast<char*>(&dim), sizeof(dim));
	fout.write(reinterpret_cast<char*>(latent_rows), sizeof(double) * local_num_rows * dim);
	fout.close();
	cp_time_write[thread_index] += (tbb::tick_count::now() - t0).seconds();
}

void NomadBody::_archive_msg_queue(int thread_index, const string& suffix, colque& queue, bool locked)
{
	tbb::tick_count t0 = tbb::tick_count::now();
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
	cp_time_write[thread_index] += (tbb::tick_count::now() - t0).seconds();
}

void NomadBody::arhive_job_queue(int thread_index, bool locked)
{
	VLOG(2) << "W" << mpi_rank << ": archive job queue";
	_archive_msg_queue(thread_index, ".msg", job_queues[thread_index], locked);
}

void NomadBody::arhive_send_queue(bool locked)
{
	VLOG(2) << "W" << mpi_rank << ": archive send queue";
	_archive_msg_queue(0, ".smsg", send_queue, locked);
}

void NomadBody::archive_msg(int thread_index, ColumnData* p_col){
	char* buffer = new char[unit_bytenum];
	tbb::tick_count t0 = tbb::tick_count::now();
	p_col->serialize(buffer, option->latent_dimension_);
	cp_fmsgs[thread_index]->write(buffer, unit_bytenum);
	cp_time_write[thread_index] += (tbb::tick_count::now() - t0).seconds();
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

/////////////////////////////////////////////////////////
// Define Checkpoint Util Functions
/////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////
// Define Checkpoint Method Specific Functions
/////////////////////////////////////////////////////////

// machine level

void NomadBody::cp_shm_start(int epoch)
{
	VLOG(1) << mpi_rank << " m start";
	cp_time_total_timer = tbb::tick_count::now();
	cp_epoch = epoch;
	checkpointing = true;
	if(option->cp_type_ == "sync"){
		allow_sending = false;
		allow_processing = false;
		_send_clear_signal(true);
	}else if(option->cp_type_ == "async"){
		_send_sig2threads_start(epoch);
		//_send_clear_signal(false);
	}else if(option->cp_type_ == "vs"){
		allow_sending = false;
		_send_clear_signal(true);
	} else{

	}
}

void NomadBody::cp_shm_clear(int epoch, int source)
{
	VLOG(1) << mpi_rank << " m clear " << source;
	LOG_IF(FATAL, epoch != cp_epoch) << "epoch of checkpoint does not match: " << cp_epoch << " vs " << epoch;
	if(option->cp_type_ == "sync"){
		++cp_received_clear_counter;
		if(cp_received_clear_counter == mpi_size){
			for(int i = 0; i < option->num_threads_; ++i)
				cp_action_ready[i] = true;
		}
	} else if(option->cp_type_ == "async"){
		_send_sig2threads_clear(source);
	} else if(option->cp_type_ == "vs"){
		++cp_received_clear_counter;
		if(cp_received_clear_counter == mpi_size){
			// temporarily stop processing to stablize the send_queue
			allow_processing = false;
			for(int i = 0; i < option->num_threads_; ++i)
				cp_action_ready[i] = true;
		}
	} else{

	}
}

void NomadBody::cp_shm_resume(int epoch)
{
	VLOG(1) << mpi_rank << " m resume";
	LOG_IF(FATAL, epoch != cp_epoch) << "epoch of checkpoint does not match: " << cp_epoch << " vs " << epoch;
	if(option->cp_type_ == "sync"){
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
	} else{

	}
	cp_received_clear_counter = 0;
	cp_ut_wait_counter = 0;
	checkpointing = false;
	cp_time_total_worker += (tbb::tick_count::now() - cp_time_total_timer).seconds();
}

// thread level

void NomadBody::cp_sht_start(int thread_index, int part_index, int epoch, double* latent_rows, int local_num_rows)
{
	VLOG(2) << mpi_rank << "-" << thread_index << " t start";
	if(option->cp_type_ == "sync"){
		// nothing
	} else if(option->cp_type_ == "async"){
		_sync_all_update_thread();
		for(int i = 0; i < mpi_size; ++i)
			cp_need_archive_msg_from[thread_index][i] = true;
		archive_local(thread_index, latent_rows, local_num_rows);
		cp_fmsgs[thread_index] = new ofstream(gen_cp_file_name(part_index) + ".msg", ofstream::binary);
		if(thread_index == 0){
			_send_clear_signal(false);
		}
	} else if(option->cp_type_ == "vs"){
		// nothing
	} else{

	}
}

void NomadBody::cp_sht_clear(int thread_index, int part_index, int source, double* latent_rows, int local_num_rows)
{
	VLOG(2) << mpi_rank << "-" << thread_index << " t clear: " << source;
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
	VLOG(1) << mpi_rank << "-" << thread_index << " cp_uf ";
	if(option->cp_type_ == "sync"){
		// triggered by the last clear signal
		_sync_all_update_thread();
		archive_local(thread_index, latent_rows, local_num_rows);
		arhive_job_queue(thread_index);
		if(thread_index == 0){
			arhive_send_queue();
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
