#include "nomad_body.h"

#include <tbb/tbb.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>

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
			send_queue_force.emplace(ColumnData::SIGNAL_CP_START, cp_master_epoch);
			// wait for local finish
			while(cp_master_lfinish_count < mpi_size){
				this_thread::sleep_for(chrono::duration<double>(0.05));
				//this_thread::yield();
			}
			send_queue_force.emplace(ColumnData::SIGNAL_CP_RESUME, cp_master_epoch);
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
