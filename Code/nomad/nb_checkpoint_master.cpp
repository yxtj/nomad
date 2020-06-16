#include "nomad_body.h"

#include <tbb/tbb.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>

#include <mpi.h>
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
			// wait for local finish
			while(cp_master_lfinish_count < numtasks){
				this_thread::sleep_for(chrono::duration<double>(0.05));
				//this_thread::yield();
			}
			for(int i = 0; i < numtasks; ++i){
				MPI_Ssend(reinterpret_cast<char*>(&cp_master_epoch), sizeof(cp_master_epoch), MPI_CHAR, i, MsgType::CP_RESUME, MPI_COMM_WORLD);
			}
			// finish
			cp_master_lfinish_count = 0;
			cp_master_epoch++;
			last_cptime = tbb::tick_count::now();
		}
	}
}

void NomadBody::cp_sh_m_lfinish(int source)
{
	++cp_master_lfinish_count;
}
