#include "nomad_body.h"
#include <chrono>
#include <iostream>

using namespace std;

void NomadBody::do_net_control_delay(){
	if(control_net_delay){
		std::this_thread::sleep_for(net_delay);
	}
}

void NomadBody::do_net_control_ratio(size_t nbyte, tbb::tick_count::interval_t time){
	if(!control_net_ratio)
		return;
	double t = nbyte / net_ratio - time.seconds();
	//cout << log_header << t << endl;
	if(t > 0){
		std::this_thread::sleep_for(std::chrono::duration<double>(t));
	}
}

