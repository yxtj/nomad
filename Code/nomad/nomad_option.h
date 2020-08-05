#ifndef OPTION_HPP_
#define OPTION_HPP_

#include <vector>
#include <string>

 //#include <boost/program_options.hpp>

struct NomadOption {
	int num_threads_;
	double learn_rate_;
	double decay_rate_;
	double par_lambda_;
	unsigned long long seed_;
	int latent_dimension_;

	std::vector<double> timeouts_;
	double stop_rmse;
	double min_error;
	double report_interval; // time interval of reporting local error
	double termcheck_min_portion; // min number of updated column to trigger termination check

	int column_per_msg;
	double interval_per_msg;
	//int pipeline_token_num_;
	int num_reuse_;
	bool flag_pause_;
	bool flag_random_send;
	double rank0_delay_;

	std::string path_;
	std::string output_path_;

	int recover_epoch;
	int job_id_;
	std::string cp_type_;
	double cp_interval_;
	std::string cp_path_;

	double net_delay;
	double net_ratio;
	std::string net_ratio_str;

	int verbose_level;

	bool parse_command(int& argc, char**& argv);

private:
	void to_lower(std::string& s);

	bool check_ctype();

	bool parse_ratio();


};

#endif
