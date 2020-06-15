#include "nomad_option.h"

#include <vector>
#include <numeric>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/format.hpp>

using namespace std;

namespace nomad {

	NomadOption::NomadOption(const char* program_name) :
		option_desc_((boost::format("%s options") % program_name).str().c_str())
	{
		using boost::program_options::value;

		option_desc_.add_options()
			("help,h", "produce help message")
			("nthreads", value<int>(&num_threads_)->default_value(4),
				"number of threads to use (0: automatic)")
			("lrate,l", value<double>(&learn_rate_)->default_value(0.001),
				"learning rate")
			("drate,d", value<double>(&decay_rate_)->default_value(0.1),
				"decay rate")
			("reg,r", value<double>(&par_lambda_)->default_value(1.0),
				"regularization parameter lambda")
			("seed,s", value<unsigned long long>(&seed_)->default_value(12345),
				"seed value of random number generator")
			("timeout,t",
				value<vector<double> >(&timeouts_)->multitoken()->default_value(vector<double>(1, 10.0), "10.0"),
				"timeout seconds until completion")
			("ptoken,p", value<int>(&pipeline_token_num_)->default_value(1024),
				"number of tokens in the pipeline")
			("dim", value<int>(&latent_dimension_)->default_value(100),
				"dimension of latent space")
			("reuse", value<int>(&num_reuse_)->default_value(1),
				"number of column reuse")
			("pause", value<bool>(&flag_pause_)->default_value(true),
				"number of column reuse")
			("r0delay", value<double>(&rank0_delay_)->default_value(0),
				"arbitrary network delay added to communication of rank 0 machine")
			("path", value<string>(&path_), "path of data")
			("output", value<string>(&output_path_)->default_value(""),
				"path of the file the result will be printed into")

			("jobid,j", value<int>(&job_id_)->default_value(1),
				"numberic ID of current job, used to distinguish different job's internal files.")
			("cptype,c", value<string>(&cp_type_)->default_value("none"),
				"type of checkpoint (none, sync, async, vs)")
			("cpinterval,i", value<double>(&cp_interval_)->default_value(-1.0),
				"interval of checkpoint (negative value disable checkpoint, by default)")
			("cppath", value<string>(&cp_path_)->default_value(""),
				"path to store checkpoint")

			("net-delay", value<double>(&net_delay)->default_value(0),
				"the additional delay in network control")
			("net-ratio", value<std::string>(&net_ratio_str)->default_value("inf"),
				"the maximum sending ratio in network control (MB/s)")
			;
		net_ratio = numeric_limits<double>::infinity();
	}

	bool NomadOption::is_option_OK()
	{
		if(path_.length() <= 0){
			cerr << "input path has to be specified." << endl;
			return false;
		}
		return true;
	}

	bool NomadOption::parse_ratio()
	{
		string s = net_ratio_str;
		for(size_t i = 0; i < s.size(); ++i) {
			if(s[i] >= 'A' && s[i] <= 'Z')
				s[i] += 'a' - 'A';
		}
		bool flag = true;
		size_t scale = 1;
		if(s.size() < 2){
			flag = false;
		} else if(s == "inf"){
			net_ratio = numeric_limits<double>::infinity();
		} else{
			if(s.back() == 'k' || s.back() == 'm' || s.back() == 'g'){
				if(s.back() == 'k')
					scale = 1000;
				else if(s.back() == 'm')
					scale = 1000 * 1000;
				else
					scale = 1000 * 1000 * 1000;
				s = s.substr(0, s.size() - 1);
			}
			try{
				net_ratio = stod(s);
			} catch(...){
				flag = false;
			}
		}
		if(flag && scale != 1){
			net_ratio *= scale;
		}
		return flag;
	}

	bool NomadOption::parse_command(int& argc, char**& argv) {

		using std::cerr;
		using std::endl;

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
				flag_help = true;
			}

		} catch(std::exception& excep) {
			cerr << "error: " << excep.what() << "\n";
			flag_help = true;
		} catch(...) {
			cerr << "Exception of unknown type!\n";
			flag_help = true;
		}

		if(true == flag_help || false == is_option_OK()) {
			cerr << option_desc_ << endl;
			return false;
		}

		return true;

	}

}
