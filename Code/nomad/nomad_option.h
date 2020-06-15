/*
 * Copyright (c) 2013 Hyokun Yun
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */
#ifndef NOMAD_NOMAD_OPTION_HPP_
#define NOMAD_NOMAD_OPTION_HPP_

//#include "nomad.hpp"

//#include <random>
#include <vector>
//#include <numeric>

#include <boost/program_options.hpp>

namespace nomad {

	using std::vector;
	using std::string;

	struct NomadOption {

		boost::program_options::options_description option_desc_;
		int num_threads_;
		double learn_rate_;
		double decay_rate_;
		double par_lambda_;
		unsigned long long seed_;
		int latent_dimension_;
		vector<double> timeouts_;
		int pipeline_token_num_;
		int num_reuse_;
		bool flag_pause_;
		double rank0_delay_;

		string path_;
		string output_path_;

		int job_id_;
		string cp_type_;
		double cp_interval_;
		string cp_path_;

		double net_delay;
		double net_ratio;
		string net_ratio_str;

		NomadOption(const char* program_name);

		virtual int get_num_cols() = 0;

		bool is_option_OK();

		bool parse_ratio();

		bool parse_command(int& argc, char**& argv);

	};

}

#endif
