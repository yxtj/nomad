/*
 * CheckpointState.hpp
 *
 *  Created on: Mar 23, 2016
 *      Author: tzhou
 */

#ifndef CHECKPOINTSTATE_H_
#define CHECKPOINTSTATE_H_

#include <vector>
#include <fstream>
#include <string>

namespace nomad{

struct CheckpointState{
	bool checkpointing=false;
	std::vector<bool> received_flush;
	int count_recv_flush=0;
	int epoch=-1;
	std::string folder;
	std::ofstream* fmsg=nullptr;
public:
	CheckpointState(){}
	CheckpointState(const int n_part):received_flush(n_part,false) {}
	static std::string get_folder(const std::string& cp_path,const int job_id,const int epoch){
		return cp_path + (cp_path.empty()?"":"/") +
				std::to_string(job_id) +"/epoch-"+ std::to_string(epoch)+"/";
	}
	static std::string get_fname_message(const int part_index){
		return std::to_string(part_index)+".msg";
	}
	static std::string get_fname_state(const int part_index){
		return std::to_string(part_index)+".state";
	}

	void start(const std::string& cp_path, const int epoch_, const int part_index);
	void archive_msg(const char* data, const int len);
	void archive_state(const char* data, const int len);

};

} // namespace nomad

#endif /* CHECKPOINTSTATE_H_ */
