/*
 * CheckpointState.cpp
 *
 *  Created on: Mar 28, 2016
 *      Author: tzhou
 */

#include "CheckpointState.h"

using namespace std;


void CheckpointState::start(const std::string& cp_path, const int epoch_, const int part_index){
	checkpointing=true;
	for(int i=0;i<received_flush.size();++i)
		received_flush[i]=false;
	count_recv_flush=0;
	epoch=epoch_;
	folder=cp_path;
	fmsg=ofstream(folder+ std::to_string(part_index)+".msg", ofstream::binary);
}


