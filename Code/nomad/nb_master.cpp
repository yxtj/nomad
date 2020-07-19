#include "nomad_body.h"

#include <iomanip>
#include <algorithm>
#include <numeric>
#include <boost/format.hpp>
#include <glog/logging.h>

using namespace std;

/////////////////////////////////////////////////////////
// Define Master Termination Chek
/////////////////////////////////////////////////////////

void NomadBody::master_termcheck()
{
	tbb::tick_count start_time = tbb::tick_count::now();
	const long long mini_stop_update = static_cast<long long>(8 * mpi_size) * global_num_cols;
	global_error = numeric_limits<double>::infinity();
	double diff = numeric_limits<double>::infinity();
	while(!finished
		&& diff > option->min_error // improvement condition
		&& (tm_global_update_count < mini_stop_update || global_error > option->stop_rmse) ) // value condition
	{
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
		diff = rmse - global_error;
		double time = (tbb::tick_count::now() - start_time).seconds();
		LOG(INFO) << boost::format("M: termination check at %.2lf: RMSE: %.12lf, difference: %g, update: %lld")
			% time % rmse % diff % tm_global_update_count_new;
		diff = abs(diff);
		global_error = rmse;
	}
	LOG(INFO) << "M: send terimination signal";
	send_queue_force.emplace(ColumnData::SIGNAL_TERMINATE, tm_count);
}

void NomadBody::sh_m_lerror(int source, double error, long long count)
{
	VLOG(1) << "M: tm receive from " << source << ": " << error << " - " << count << endl;
	{
		std::unique_lock<std::mutex> lk(tm_m);
		tm_local_error_received[source] = error;
		tm_local_error_ready[source] = true;
		tm_local_update_count[source] = count;
	}
	if(all_of(tm_local_error_ready, tm_local_error_ready + mpi_size,
		[](const atomic<bool>& b){ return b.load(); }))
	{
		DVLOG(1) << "M: tm notify";
		tm_cv.notify_all();
	}
}

/////////////////////////////////////////////////////////
// Define Master Checkpoint
/////////////////////////////////////////////////////////

void NomadBody::master_checkpoint(){
	tbb::tick_count start_time = tbb::tick_count::now();
	LOG(INFO) << "M: start checkpoint thread" << endl;
	tbb::tick_count last_cptime = start_time;
	std::chrono::duration<double> cp_interval(option->cp_interval_);
	tbb::tick_count cp_start_time;
	while(!finished){
		std::unique_lock<std::mutex> lk(cp_m);
		// use cp_cv.wait_for to implement an interruptable sleep_for
		bool btm = cp_cv.wait_for(lk, cp_interval, [&](){
			return (tbb::tick_count::now() - last_cptime).seconds() >= option->cp_interval_;
			});
		if(!finished && flag_train_ready && !flag_train_stop){
			if(!btm)
				continue;
			cp_start_time = tbb::tick_count::now();
			LOG(INFO) << "M: start checkpoint " << cp_master_epoch << " at " << (cp_start_time - start_time).seconds();
			send_queue_force.emplace(ColumnData::SIGNAL_CP_START, cp_master_epoch);
			// wait for local finish
			while(cp_master_lfinish_count < mpi_size){
				this_thread::sleep_for(chrono::duration<double>(0.05));
				//this_thread::yield();
			}
			send_queue_force.emplace(ColumnData::SIGNAL_CP_RESUME, cp_master_epoch);
			// finish
			cp_master_lfinish_count = 0;
			last_cptime = tbb::tick_count::now();
			double duration = (last_cptime - cp_start_time).seconds();
			cp_time_total_master += duration;
			LOG(INFO) << "M: finish checkpoint " << cp_master_epoch << " at " << (last_cptime - start_time).seconds()
				<< " duration: " << duration;
			cp_master_epoch++;
		}
	}
	LOG(INFO) << "M: finish checkpoint thread";
}

void NomadBody::cp_sh_m_lfinish(int source)
{
	++cp_master_lfinish_count;
}
