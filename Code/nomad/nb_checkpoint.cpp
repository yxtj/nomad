#include "nomad_body.h"

#include <tbb/tbb.h>
//#define TBB_IMPLEMENT_CPP0X
//#include <tbb/compat/thread>

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32


using namespace std;

void NomadBody::master_func(){
	string log_header = "M: ";
	cout << log_header << "master thread start" << endl;
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
			cout << log_header << "sending out checkpoint signal " << cp_master_epoch << endl;
			for(int i = 0; i < numtasks; ++i){
				MPI_Ssend(reinterpret_cast<char*>(&cp_master_epoch), sizeof(cp_master_epoch), MPI_CHAR, i, MsgType::CP_START, MPI_COMM_WORLD);
			}
			cp_master_epoch++;
			last_cptime = tbb::tick_count::now();
		}
	}
}

/////////////////////////////////////////////////////////
// Define Checkpoint functions
/////////////////////////////////////////////////////////
void NomadBody::signal_handler_start(int thread_index, ColumnData* p_col, double* latent_rows, int local_num_rows, int dim){
	int epoch = p_col->pos_;
	int part_index = rank * option->num_threads_ + thread_index;
	//		cerr<<"starting cp "<<epoch<<" at "<<part_index<<endl;
	if(start_cp(thread_index, epoch)){
		cout << log_header << "checkpoint start: " << cp_epoch[thread_index] << " on " << rank << "," << thread_index << endl;
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
void NomadBody::signal_handler_flush(int thread_index, ColumnData* p_col){
	//		int part_index = rank * option->num_threads_ + thread_index;
	int source = p_col->pos_;
	//		cerr<<"receive flush signal at "<<part_index<<" from "<<source<<endl;
	received_flush[thread_index][source] = true;
	++count_recv_flush[thread_index];
	if(count_recv_flush[thread_index] == num_parts){
		checkpointing[thread_index] = false;
		finish_cp(thread_index);
		cout << log_header << "checkpoint finish: " << cp_epoch[thread_index] << " on " << rank << "," << thread_index << endl;
	}
}

bool NomadBody::start_cp(int thread_index, int epoch){
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

void NomadBody::archive_local(int thread_index, double* latent_rows, int local_num_rows, int dim){
	int part_index = rank * option->num_threads_ + thread_index;
	tick_count t0 = tbb::tick_count::now();
	ofstream fout(cp_folder + std::to_string(part_index) + ".state", ofstream::binary);
	fout.write(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
	fout.write(reinterpret_cast<char*>(&dim), sizeof(dim));
	fout.write(reinterpret_cast<char*>(latent_rows), sizeof(double) * local_num_rows * dim);
	fout.close();
	cp_write_time[thread_index] += (tbb::tick_count::now() - t0).seconds();
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
	cp_fmsgs[thread_index]->close();
	cp_fmsgs[thread_index]->clear();
	delete cp_fmsgs[thread_index];
	checkpointing[thread_index] = false;
	//cerr<<"Message archived at "<<cp_epoch[thread_index]<<" on "<<rank<<","<<thread_index
	//		<<" : "<<msg_archived[thread_index]<<" . queue-len : "<<job_queues[thread_index].unsafe_size()<<endl;
}

void NomadBody::restore_local(const string& cp_folder, int part_index, double* latent_rows, int& local_num_rows, int& dim){
	ifstream fin(cp_folder + std::to_string(part_index) + ".state", ofstream::binary);
	fin.read(reinterpret_cast<char*>(&local_num_rows), sizeof(local_num_rows));
	fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
	fin.read(reinterpret_cast<char*>(latent_rows), sizeof(double) * local_num_rows * dim);
	fin.close();
}

void NomadBody::restore_msg(const string& cp_folder, int part_index){
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

void NomadBody::restore(int epoch, int thread_index, double* latent_rows, int& local_num_rows, int& dim){
	cp_epoch[thread_index] = epoch;
	int part_index = rank * option->num_threads_ + thread_index;
	cp_folder = option->cp_path_ + (option->cp_path_.empty() ? "" : "/") +
		std::to_string(option->job_id_) + "/epoch-" + std::to_string(cp_epoch[thread_index]) + "/";
	restore_local(cp_folder, part_index, latent_rows, local_num_rows, dim);
	restore_msg(cp_folder, part_index);
}


