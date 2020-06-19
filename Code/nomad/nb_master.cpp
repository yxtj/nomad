#include "nomad_body.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <boost/format.hpp>

using namespace std;

using nomad::ColumnData;
using nomad::MsgType;

/////////////////////////////////////////////////////////
// Define Master Termination Chek
/////////////////////////////////////////////////////////

void NomadBody::master_termcheck()
{
	tbb::tick_count start_time = tbb::tick_count::now();
	global_error = numeric_limits<double>::infinity();
	double diff = numeric_limits<double>::infinity();
	while(!finished && diff > option->min_error){
		std::unique_lock<std::mutex> lk(tm_m);
		tm_cv.wait(lk);
		if(finished)
			break;
		for(int i = 0; i < mpi_size; ++i)
			tm_local_error_ready[i] = false;
		long long tm_global_update_count_new = accumulate(tm_local_update_count.begin(), tm_local_update_count.end(), 0ll);
		if(tm_global_update_count_new - tm_global_update_count < tm_min_updated_col)
			continue;
		tm_global_update_count = tm_global_update_count_new;
		double sum = accumulate(tm_local_error_received.begin(), tm_local_error_received.end(), 0.0);
		double rmse = sqrt(sum / global_num_nonzero);
		diff = abs(rmse - global_error);
		double time = (tbb::tick_count::now() - start_time).seconds();
		cout << boost::format("M: termination check at %.2lf: last RMSE: %g new RMSE: %g difference: %g")
			% time % global_error % rmse % diff << endl;
		global_error = rmse;
	}
	cout << "M: send terimination signal" << endl;
	send_queue_force.emplace(ColumnData::SIGNAL_TERMINATE, tm_count);
}

void NomadBody::sh_m_lerror(int source, double error, long long count)
{
	cout << "M: tm receive: " << error << " - " << count << endl;
	{
		std::unique_lock<std::mutex> lk(tm_m);
		tm_local_error_received[source] = error;
		tm_local_error_ready[source] = true;
		tm_local_update_count[source] = count;
	}
	if(all_of(tm_local_error_ready, tm_local_error_ready + mpi_size,
		[](const atomic<bool>& b){ return b.load(); }))
	{
		cout << "M: tm notify" << endl;
		tm_cv.notify_all();
	}
}

/////////////////////////////////////////////////////////
// Define Master Checkpoint
/////////////////////////////////////////////////////////

void NomadBody::master_checkpoint(){
	tbb::tick_count start_time = tbb::tick_count::now();
	cout << "M: start checkpoint thread" << endl;
	tick_count last_cptime = start_time;
	std::chrono::duration<double> cp_interval(option->cp_interval_);
	while(!finished){
		std::unique_lock<std::mutex> lk(cp_m);
		// use cp_cv.wait_for to implement an interruptable sleep_for
		bool btm = cp_cv.wait_for(lk, cp_interval, [&](){
			return (tbb::tick_count::now() - last_cptime).seconds() >= option->cp_interval_;
			});
		if(!finished && flag_train_ready && !flag_train_stop){
			if(!btm)
				continue;
			cout << "M: start checkpoint " << cp_master_epoch << " at " << (tbb::tick_count::now() - start_time).seconds() << endl;
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
			cout << "M: finish checkpoint " << cp_master_epoch << " at " << (last_cptime - start_time).seconds() << endl;
		}
	}
}

void NomadBody::cp_sh_m_lfinish(int source)
{
	++cp_master_lfinish_count;
}
