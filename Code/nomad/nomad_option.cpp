#include "nomad_option.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <boost/program_options.hpp>

using namespace std;

bool NomadOption::parse_command(int& argc, char**& argv) {
	boost::program_options::options_description option_desc_("nomad options");
	using boost::program_options::value;

	option_desc_.add_options()
		("help,h", "produce help message")
		("nthreads,n", value<int>(&num_threads_)->default_value(4),
			"number of threads to use (0: automatic)")
		("lrate,l", value<double>(&learn_rate_)->default_value(0.001),
			"learning rate")
		("drate,d", value<double>(&decay_rate_)->default_value(0.1, "0.1"),
			"decay rate")
		("reg,r", value<double>(&par_lambda_)->default_value(1.0),
			"regularization parameter lambda")
		("seed,s", value<unsigned long long>(&seed_)->default_value(12345),
			"seed value of random number generator")
		("cpm", value<int>(&column_per_msg)->default_value(100),
			"maximum number of columns buffered in a message")
		("tpm", value<double>(&interval_per_msg)->default_value(0.5),
			"maximum time interval between two consecutive messages")
		("timeout,t",
			value<vector<double> >(&timeouts_)->multitoken()->default_value(vector<double>(1, 10.0), "10.0"),
			"timeout seconds until completion")
		("tm_value,a", value<double>(&stop_rmse)->default_value(1, "1"),
			"the RMSE to stop training")
		("tm_error,e", value<double>(&min_error)->default_value(1e-6, "1e-6"),
			"the minimal error to stop")
		("tm_interval", value<double>(&report_interval)->default_value(1),
			"interval of reporting local error for termination check")
		("tm_portion", value<double>(&termcheck_min_portion)->default_value(0.01),
			"min portion of updated columns to do termination check")
		//("ptoken,p", value<int>(&pipeline_token_num_)->default_value(1024),
		//	"number of tokens in the pipeline")
		("dim", value<int>(&latent_dimension_)->default_value(100),
			"dimension of latent space")
		("reuse", value<int>(&num_reuse_)->default_value(1),
			"number of column reuse")
		("pause", value<bool>(&flag_pause_)->default_value(true),
			"pause for a while after each timeout")
		("random_send", value<bool>(&flag_random_send)->default_value(true),
			"send columns to random a worker")
		("r0delay", value<double>(&rank0_delay_)->default_value(0),
			"arbitrary network delay added to communication of rank 0 machine")
		("input,i", value<string>(&path_), "path of data")
		("output,o", value<string>(&output_path_)->default_value(""),
			"path of the file the result will be printed into")

		("jobid,j", value<int>(&job_id_)->default_value(1),
			"numberic ID of current job, used to distinguish different job's internal files.")
		("cptype,c", value<string>(&cp_type_)->default_value("none"),
			"type of checkpoint (none, sync, async, vs)")
		("cpinterval,k", value<double>(&cp_interval_)->default_value(0.0),
			"interval of checkpoint")
		("cppath,p", value<string>(&cp_path_)->default_value(""),
			"path to store checkpoint")

		("net_delay", value<double>(&net_delay)->default_value(0),
			"the additional delay in network control")
		("net_ratio", value<std::string>(&net_ratio_str)->default_value("inf"),
			"the maximum sending ratio in network control (MB/s)")

		("verbose,v", value<int>(&verbose_level)->default_value(0),
			"the verbose level for logging")
	;

	bool flag_help = false;

	try {

		boost::program_options::variables_map var_map;
		boost::program_options::store(
			boost::program_options::parse_command_line(argc, argv, option_desc_), var_map);
		notify(var_map);

		if(var_map.count("help")) {
			flag_help = true;
		}
		if(!parse_ratio()) {
			cerr << "net-ratio is invalid." << endl;
			flag_help = true;
		}
		if(path_.length() <= 0){
			cerr << "input path has to be specified." << endl;
			flag_help = true;
		}
		if(!check_ctype()){
			cerr << "checkpoint type is inalid." << endl;
			flag_help = true;
		}
		if(cp_type_ != "none" && cp_interval_ <= 0){
			cerr << "checkpoint interval is invalid." << endl;
			flag_help = true;
		}
		if(cp_type_ != "none" && cp_path_.empty()){
			cerr << "checkpoint path is not given." << endl;
			flag_help = true;
		}

	} catch(std::exception& excep) {
		cerr << "error: " << excep.what() << "\n";
		flag_help = true;
	} catch(...) {
		cerr << "Exception of unknown type!\n";
		flag_help = true;
	}

	if(true == flag_help) {
		cerr << option_desc_ << endl;
		return false;
	}
	return true;
}

void NomadOption::to_lower(std::string& s)
{
	for(size_t i = 0; i < s.size(); ++i) {
		if(s[i] >= 'A' && s[i] <= 'Z')
			s[i] += 'a' - 'A';
	}
}

bool NomadOption::check_ctype()
{
	to_lower(cp_type_);
	static vector<string> valid = { "none","sync","async","vs" };
	return find(valid.begin(), valid.end(), cp_type_) != valid.end();
}

bool NomadOption::parse_ratio()
{
	to_lower(net_ratio_str);
	bool flag = true;
	size_t scale = 1;
	if(net_ratio_str.size() < 2){
		flag = false;
	} else if(net_ratio_str == "inf"){
		net_ratio = numeric_limits<double>::infinity();
	} else{
		if(net_ratio_str.back() == 'k' || net_ratio_str.back() == 'm' || net_ratio_str.back() == 'g'){
			if(net_ratio_str.back() == 'k')
				scale = 1000;
			else if(net_ratio_str.back() == 'm')
				scale = 1000 * 1000;
			else
				scale = 1000 * 1000 * 1000;
			net_ratio_str = net_ratio_str.substr(0, net_ratio_str.size() - 1);
		}
		try{
			net_ratio = stod(net_ratio_str);
		} catch(...){
			flag = false;
		}
	}
	if(flag && scale != 1){
		net_ratio *= scale;
	}
	return flag;
}
