#include "nomad_body.h"

#include <tbb/tbb.h>

#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include <boost/format.hpp>
#include <glog/logging.h>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

using namespace std;

/////////////////////////////////////////////////////////
// Define Message Sending/Broadcasting Functions
/////////////////////////////////////////////////////////
void NomadBody::_send_data(char* send_message, const int cur_num, const int target_rank){
	*(reinterpret_cast<int8_t*>(send_message)) = MsgType::DATA;
	*(reinterpret_cast<int*>(send_message + 1)) = static_cast<int>(send_queue.unsafe_size());
	*(reinterpret_cast<int*>(send_message + 1) + 1) = cur_num;
	int bytes = cur_num == column_per_msg ? msg_bytenum : msg_head_bytenum + cur_num * unit_bytenum;
	//LOG(INFO) << "W" << mpi_rank << "-S: data " << cur_num << " columns to " << target_rank;
	//tbb::tick_count t = tbb::tick_count::now();
	//LOG(INFO) << log_header << "s: " << *(reinterpret_cast<int*>(send_message + 1)) << " - " << *(reinterpret_cast<int*>(send_message + 1) + 1);
	MPI_Ssend(send_message, bytes, MPI_CHAR, target_rank, 0, MPI_COMM_WORLD);
	//do_net_control_ratio(msg_bytenum, tbb::tick_count::now() - t);
	local_send_count += cur_num;
}

void NomadBody::_send_lerror(const double lerror, const long long nupdate)
{
	char data[1 + sizeof(double) + sizeof(long long)];
	*reinterpret_cast<int8_t*>(data) = MsgType::LOCAL_ERROR;
	*reinterpret_cast<double*>(data + 1) = lerror;
	*reinterpret_cast<long long*>(data + 1 + sizeof(double)) = nupdate;
	//LOG(INFO) << "W" << mpi_rank << "-S: lerror " << tm_col_error_sum << " - " << nupdate;
	MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void NomadBody::_bcast_dying()
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::LOCAL_DYING;
	*reinterpret_cast<int*>(data + 1) = -(mpi_rank + 1);
	for(int i = 0; i < mpi_size; ++i){
		VLOG(1) << "W" << mpi_rank << "-S: dying to " << i;
		MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, i, 0, MPI_COMM_WORLD);
	}
}

void NomadBody::_bcast_termination(const int epoch)
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::TERMINATION;
	*reinterpret_cast<int*>(data + 1) = epoch;
	for(int i = 0; i < mpi_size; ++i)
		MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, i, 0, MPI_COMM_WORLD);
}

void NomadBody::_bcast_cp_start(const int epoch)
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::CP_START;
	*reinterpret_cast<int*>(data + 1) = epoch;
	for(int i = 0; i < mpi_size; ++i)
		MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, i, 0, MPI_COMM_WORLD);
}

void NomadBody::_bcast_cp_clear(const int epoch)
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::CP_CLEAR;
	*reinterpret_cast<int*>(data + 1) = epoch;
	for(int i = 0; i < mpi_size; ++i)
		MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, i, 0, MPI_COMM_WORLD);
}

void NomadBody::_send_cp_lfinish(const int epoch)
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::CP_LFINISH;
	*reinterpret_cast<int*>(data + 1) = epoch;
	MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void NomadBody::_bcast_cp_resume(const int epoch)
{
	char data[1 + sizeof(int)];
	*reinterpret_cast<int8_t*>(data) = MsgType::CP_RESUME;
	*reinterpret_cast<int*>(data + 1) = epoch;
	for(int i = 0; i < mpi_size; ++i)
		MPI_Ssend(reinterpret_cast<void*>(&data), sizeof(data), MPI_CHAR, i, 0, MPI_COMM_WORLD);
}

/////////////////////////////////////////////////////////
// Define Training Sender Thread
/////////////////////////////////////////////////////////
void NomadBody::train_send_func(const double timeout){
	string log_header = "W" + to_string(mpi_rank) + "-S: ";
	mt19937_64 send_rng(mpi_rank * 17 + option->seed_ + option->num_threads_ + 2);
	std::uniform_int_distribution<> target_dist(0, mpi_size - 1);

	const int dim = option->latent_dimension_;

	while(flag_train_ready == false){
		std::this_thread::yield();
	}

	const tbb::tick_count start_time = tbb::tick_count::now();
	double last_send_time = 0.0;
	double last_tm_send_time = 0.0;
	long long last_tm_send_update = local_send_count;

	char* send_message = sallocator<char>().allocate(msg_bytenum);
	char* cur_pos = send_message + msg_head_bytenum;
	int cur_num = 0;

	pair<int, int> force_sent_signal;

	// Buffer some columns for one message. (# column>UNITS_PER_MSG || wait time>timeout)
	while(true){
		double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
		if((finished || elapsed_seconds > timeout) && !checkpointing){
			break;
		}

		if(send_queue_force.try_pop(force_sent_signal)){
			DLOG(INFO) << log_header << " force " << force_sent_signal.first << " - " << force_sent_signal.second;
			switch(force_sent_signal.first){
			case ColumnData::SIGNAL_LERROR:
				// to master
				last_tm_send_time = elapsed_seconds;
				last_tm_send_update = local_send_count;
				_send_lerror(tm_col_error_sum, local_send_count);
				break;
			case ColumnData::SIGNAL_TERMINATE:
				// from master
				_bcast_termination(force_sent_signal.second);
				break;
			case ColumnData::SIGNAL_CP_START:
				// from master
				_bcast_cp_start(force_sent_signal.second);
				break;
			case ColumnData::SIGNAL_CP_CLEAR:
				// between workers
				_bcast_cp_clear(force_sent_signal.second);
				break;
			case ColumnData::SIGNAL_CP_LFINISH:
				// between workers
				_send_cp_lfinish(force_sent_signal.second);
				break;
			case ColumnData::SIGNAL_CP_RESUME:
				// from master
				_bcast_cp_resume(force_sent_signal.second);
				break;
			default:
				LOG(WARNING) << log_header << "Unsupported signal type: " << force_sent_signal.first;
				break;
			}
			continue;
		}

		ColumnData* p_col = nullptr;

		if(allow_sending && send_queue.try_pop(p_col)){
			// CP related
			if(p_col->col_index_ == ColumnData::SIGNAL_CP_CLEAR){
				//flush out out-message
				if(cur_num != 0){
					int target_rank = target_dist(send_rng);
					_send_data(send_message, cur_num, target_rank);
					cur_pos = send_message + msg_head_bytenum;
					cur_num = 0;
				}
				//send clear signal
				_bcast_cp_clear(p_col->pos_);
				column_pool->push(p_col);
				continue;
			}

			p_col->serialize(cur_pos, dim);
			// incrementally update tm_col_error_sum
			tm_col_error_sum += p_col->error - tm_col_error[p_col->col_index_];
			tm_col_error[p_col->col_index_] = p_col->error;

			column_pool->push(p_col);
			cur_pos += unit_bytenum;
			cur_num++;

			if(cur_num >= column_per_msg){
				int target_rank = option->flag_random_send ? target_dist(send_rng) : ((mpi_rank + 1) % mpi_size);
				//LOG(INFO) << mpi_rank << " send";
				_send_data(send_message, cur_num, target_rank);
				cur_pos = send_message + msg_head_bytenum;
				cur_num = 0;
			}
			// send local error for termination check
			if(elapsed_seconds - last_tm_send_time > option->report_interval && last_tm_send_update != local_send_count)
			{
				last_tm_send_time = elapsed_seconds;
				last_tm_send_update = local_send_count;
				_send_lerror(tm_col_error_sum, local_send_count);
			}

		} else{
			// fail in send_queue.try_pop(p_col)
			if(cur_num > 0 && elapsed_seconds - last_send_time > option->interval_per_msg){
				last_send_time = elapsed_seconds;
				int target_rank = target_dist(send_rng);
				_send_data(send_message, cur_num, target_rank);
				cur_pos = send_message + msg_head_bytenum;
				cur_num = 0;
			}
			std::this_thread::yield();
		}

	} // elapsed_seconds > timeout

	// send remaining columns to random machine
	if(cur_num != 0){
		int target_rank = target_dist(send_rng);
		_send_data(send_message, cur_num, target_rank);
	}

	// send dying message to every machine
	LOG(INFO) << log_header << "send dying message";
	_bcast_dying();

	sallocator<char>().deallocate(send_message, msg_bytenum);
	LOG(INFO) << log_header << "send thread finishing";
} //end of train_send_func

/////////////////////////////////////////////////////////
// Define Training Receive Function
/////////////////////////////////////////////////////////
void NomadBody::train_recv_func(){
	string log_header = "W" + to_string(mpi_rank) + "-R: ";
	const int dim = option->latent_dimension_;
	char* recv_message = sallocator<char>().allocate(msg_bytenum);

	//const tick_count start_time = tick_count::now();

	int num_dead = 0;

	MPI_Status status;

	while(num_dead < mpi_size){
		int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		//do_net_control_delay();
		LOG_IF(FATAL, rc != MPI_SUCCESS) << "ReceiveTask MPI Error";

		int8_t mtype = *reinterpret_cast<int8_t*>(recv_message);
		DLOG_IF(INFO, mtype != MsgType::DATA) << log_header << int(mtype);
		switch(mtype)
		{
		case MsgType::DATA:{
			int queue_size = *(reinterpret_cast<int*>(recv_message + 1));
			int num_received = *(reinterpret_cast<int*>(recv_message + 1) + 1);
			//LOG(INFO) << log_header << "r: " << queue_size << " - " << num_received;
			//queue_current_sizes[status.MPI_SOURCE] = queue_size;

			// negative numbers are dying messages
			char* cur_pos = recv_message + msg_head_bytenum;
			int source = status.MPI_SOURCE;// * option->num_threads_;
			for(int i = 0; i < num_received; i++){
				ColumnData* p_col = column_pool->pop();
				p_col->deserialize(cur_pos, dim);
				p_col->source_ = source;

				// generate permutation
				p_col->set_perm(option->num_threads_, rng);

				job_queues[p_col->perm_[0]].push(p_col);
				cur_pos += unit_bytenum;
			}
			break;
		}
		case MsgType::LOCAL_ERROR:{
			int source = status.MPI_SOURCE;
			double local_error_sum = *reinterpret_cast<double*>(recv_message + 1);
			long long local_update_count = *reinterpret_cast<long long*>(recv_message + 1 + sizeof(double));
			sh_m_lerror(source, local_error_sum, local_update_count);
			break;
		}
		case MsgType::LOCAL_DYING:{
			num_dead++;
			break;
		}
		case MsgType::TERMINATION:
			flag_train_stop = true;
			finished = true;
			break;
		case MsgType::CP_START:{
			int epoch = *reinterpret_cast<int*>(recv_message + 1);
			cp_shm_start(epoch);
			break;
		}
		case MsgType::CP_CLEAR:{
			int epoch = *reinterpret_cast<int*>(recv_message + 1);
			int source = status.MPI_SOURCE;
			cp_shm_clear(epoch, source);
			break;
		}
		case MsgType::CP_LFINISH:{
			// for master only
			int epoch = *reinterpret_cast<int*>(recv_message + 1);
			cp_sh_m_lfinish(epoch);
			break;
		}
		case MsgType::CP_RESUME:{
			int epoch = *reinterpret_cast<int*>(recv_message + 1);
			cp_shm_resume(epoch);
			break;
		}
		case MsgType::CP_RESTORE:{
			break;
		}
		default:
			break;
		}
	}
	sallocator<char>().deallocate(recv_message, msg_bytenum);
	LOG(INFO) << log_header << "receive thread finishing";
} // end receiving for train

/////////////////////////////////////////////////////////
// Define Testing Sender Thread
/////////////////////////////////////////////////////////
void NomadBody::test_send_func(){
	string log_header = "W" + to_string(mpi_rank) + "-S: ";
	const long mask = (1L << mpi_rank);

	const int dim = option->latent_dimension_;

	//const tick_count start_time = tick_count::now();

	char* send_message = sallocator<char>().allocate(msg_bytenum);
	char* cur_pos = send_message + sizeof(int);
	int cur_num = 0;

	int send_count = 0;

	int target_rank = mpi_rank + 1;
	target_rank %= mpi_size;

	while(send_count < global_num_cols){
		ColumnData* p_col;

		if(send_queue.try_pop(p_col)){
			// if the column was not already processed
			if((p_col->flag_ & mask) == 0){
				p_col->flag_ |= mask;
				p_col->serialize(cur_pos, dim);
				cur_pos += unit_bytenum;
				cur_num++;

				send_count++;

				if(cur_num >= column_per_msg){
					*(reinterpret_cast<int*>(send_message)) = cur_num;

					// choose destination
					int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);
					LOG_IF(FATAL, rc != MPI_SUCCESS) << "SendTask MPI Error";

					cur_pos = send_message + sizeof(int);
					cur_num = 0;
				}
			} else{
				LOG(INFO) << log_header << "!!! should not happen! flag:" << p_col->flag_ << "???";
			}
			column_pool->push(p_col);

		} else{
			// even if pop was failed, if there is remaining message send it to another machine
			if(cur_num > 0){
				*(reinterpret_cast<int*>(send_message)) = cur_num;
				// choose destination
				int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, target_rank, MsgType::DATA, MPI_COMM_WORLD);
				LOG_IF(FATAL, rc != MPI_SUCCESS) << "SendTask MPI Error";

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
		LOG_IF(FATAL, rc != MPI_SUCCESS) << "SendTask MPI Error";
	}
	sallocator<char>().deallocate(send_message, msg_bytenum);

	LOG(INFO) << log_header << "test send thread finishing";
} // end of test_send_func

/////////////////////////////////////////////////////////
// Define Testing Receive Function
/////////////////////////////////////////////////////////
void NomadBody::test_recv_func(){
	string log_header = "W" + to_string(mpi_rank) + "-R: ";
	const int dim = option->latent_dimension_;
	char* recv_message = sallocator<char>().allocate(msg_bytenum);

	//const tick_count start_time = tick_count::now();

	int recv_count = 0;

	MPI_Status status;

	const long mask = (1L << mpi_rank);

	while(recv_count < global_num_cols){
		int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		LOG_IF(FATAL, rc != MPI_SUCCESS) << "ReceiveTask MPI Error";
		if(status.MPI_TAG != MsgType::DATA){
			LOG(INFO) << log_header << "received a signal while testing. signal " << status.MPI_TAG << ", from " << status.MPI_SOURCE;
			continue;
		}

		int num_received = *(reinterpret_cast<int*>(recv_message));
		// negative numbers are dying messages
		char* cur_pos = recv_message + sizeof(int);
		for(int i = 0; i < num_received; i++){

			ColumnData* p_col = column_pool->pop();
			p_col->deserialize(cur_pos, dim);

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

