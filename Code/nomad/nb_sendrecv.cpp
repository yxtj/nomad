#include "nomad_body.h"

#include <tbb/tbb.h>
//#define TBB_IMPLEMENT_CPP0X
//#include <tbb/compat/thread>

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include <boost/format.hpp>

#include "mpi.h"
#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

using namespace std;

/////////////////////////////////////////////////////////
// Define Training Sender Thread
/////////////////////////////////////////////////////////
void NomadBody::train_send_func(const double timeout){
	string log_header = "W" + to_string(rank) + "-S: ";
	mt19937_64 send_rng(rank * 17 + option->seed_ + option->num_threads_ + 2);
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

		//if(monitor_num < elapsed_seconds){
		//	cout << log_header << "sender thread alive," << rank << "," << monitor_num << ","
		//			<< send_queue.unsafe_size() << ",endline" << endl;
		//	monitor_num++;
		//}

		ColumnData* p_col = nullptr;

		if(allow_sending && send_queue.try_pop(p_col)){
			// CP related
			if(p_col->col_index_ == ColumnData::SIGNAL_CP_CLEAR){
				//flush out out-message
				if(cur_num != 0){
					int target_rank = target_dist(send_rng);
					_send_msg(send_message, cur_num, target_rank);

					local_send_count += cur_num;
					cur_pos = send_message + sizeof(int) + sizeof(int);
					cur_num = 0;
				}
				//send clear signal
				*reinterpret_cast<int*>(send_message) = p_col->pos_;	//set source part_index
				for(int target_rank = 0; target_rank < numtasks; ++target_rank){
					if(target_rank == rank)
						continue;
					int rc = MPI_Ssend(send_message, sizeof(int), MPI_CHAR, target_rank, MsgType::CP_CLEAR, MPI_COMM_WORLD);
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

	cout << log_header << "send thread finishing" << endl;
} //end of train_send_func

/////////////////////////////////////////////////////////
// Define Training Receive Function
/////////////////////////////////////////////////////////
void NomadBody::train_recv_func(){
	string log_header = "W" + to_string(rank) + "-R: ";
	const int dim = option->latent_dimension_;
	char* recv_message = sallocator<char>().allocate(msg_bytenum);

	//const tick_count start_time = tick_count::now();
	//int monitor_num = 0;

	int num_dead = 0;

	MPI_Status status;

	while(num_dead < numtasks){

		//double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
		//if(monitor_num < elapsed_seconds){
		//	cout << log_header << "receiver thread alive," << rank << "," << monitor_num << endl;
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
			int epoch = *reinterpret_cast<int*>(recv_message);
			cp_shm_start(epoch);
		} else if(status.MPI_TAG == MsgType::CP_CLEAR){
			int source = *reinterpret_cast<int*>(recv_message);
			cp_shm_clear(source);
		} else if(status.MPI_TAG == MsgType::CP_RESUME){
			int epoch = *reinterpret_cast<int*>(recv_message);
			cp_shm_resume(epoch);
		} else if(status.MPI_TAG == MsgType::CP_LFINISH){ // for master
			int epoch = *reinterpret_cast<int*>(recv_message);
			cp_sh_m_lfinish(epoch);
		}
	}
	sallocator<char>().deallocate(recv_message, msg_bytenum);
} // end receiving for train

/////////////////////////////////////////////////////////
// Define Testing Sender Thread
/////////////////////////////////////////////////////////
void NomadBody::test_send_func(){
	string log_header = "W" + to_string(rank) + "-S: ";
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
		//				cout << log_header << "test sender thread alive: " << monitor_num << endl;
		//				monitor_num++;
		//			}

		ColumnData* p_col;

		if(send_queue.try_pop(p_col)){
			// if the column was not already processed
			if((p_col->flag_ & mask) == 0){
				p_col->flag_ |= mask;
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
				cout << log_header << "!!! should not happen! flag:" << p_col->flag_ << "???" << endl;
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

	cout << log_header << "test send thread finishing," << rank << endl;

} // end of test_send_func

/////////////////////////////////////////////////////////
// Define Testing Receive Function
/////////////////////////////////////////////////////////
void NomadBody::test_recv_func(){
	string log_header = "W" + to_string(rank) + "-R: ";
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
		//				cout << log_header << "receiver thread alive: "<< monitor_num << endl;
		//				monitor_num++;
		//			}

		int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if(rc != MPI_SUCCESS){
			std::cerr << "ReceiveTask MPI Error" << std::endl;
			exit(64);
		}
		if(status.MPI_TAG != MsgType::DATA){
			cout << log_header << "received a signal while testing. signal " << status.MPI_TAG << ", from " << status.MPI_SOURCE << endl;
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
			//				double *dest = reinterpret_cast<double *>(cur_pos + sizeof(int)+ sizeof(long));
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

