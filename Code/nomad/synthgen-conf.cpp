/*
 * synth-conf-gen.cpp
 *
 *  Created on: Mar 24, 2016
 *      Author: tzhou
 */

#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include "pl-dis.hpp"

#include <boost/program_options.hpp>

using namespace std;

int main(int argc, char **argv){

	int num_rows, num_cols;
	string row_filename, col_filename;
	string separator;
	int seed;

	boost::program_options::options_description option_desc("synthgen-conf options");
	option_desc.add_options()
	("help,h", "produce help message")
	("seed", boost::program_options::value<int>(&seed)->default_value(12345),
			"RNG seed")
	("nrow", boost::program_options::value<int>(&num_rows)->default_value(480189),
			"number of rows")
	("ncol", boost::program_options::value<int>(&num_cols)->default_value(17770),
			"number of columns")
	("sep", boost::program_options::value<string>(&separator)->default_value(","),
			"special string to separate key and value")
	("rowfile",
			boost::program_options::value<string>(&row_filename)->default_value(
					"Results/synth/rowdegree.txt"),
			"location of row degree frequencies file")
	("colfile",
			boost::program_options::value<string>(&col_filename)->default_value(
					"Results/synth/coldegree.txt"),
			"location of column degree frequencies file");

	bool flag_help = false;

	try{

		boost::program_options::variables_map var_map;
		boost::program_options::store(
				boost::program_options::parse_command_line(argc, argv, option_desc),var_map);
		boost::program_options::notify(var_map);

		if(var_map.count("help")){
			flag_help = true;
		}

	}
	catch(std::exception& excep){
		cerr << "error: " << excep.what() << "\n";
		flag_help = true;
	}
	catch(...){
		cerr << "Exception of unknown type!\n";
		flag_help = true;
	}

	if(true == flag_help){
		cerr << option_desc << endl;
		return 1;
	}

	mt19937 rng(seed);

	// row degree distribution:
	normal_distribution<> dis_row(20,5);
	auto gen_row=[&](){
		int v=static_cast<int>(dis_row(rng));
		while(v<0 || v>=num_cols){
			v=static_cast<int>(dis_row(rng));
		}
		return v;
	};
	{
		ofstream row_file(row_filename);
		cout << "writing file: " << row_filename << endl;

		if(!row_file.is_open()){
			cerr << "could not open: " << row_filename << endl;
			return 1;
		}

		// maximum degree of row is num_col
		for(int i=0;i<=num_cols;++i){
			row_file<<i<<separator<<gen_row()<<"\n";
		}
	}

	// column degree distribution:
	power_law_distribution<> dis_col(1,2.2);
	auto gen_col=[&](){
		int v=static_cast<int>(dis_col(rng));
		while(v>num_rows){
			v=static_cast<int>(dis_col(rng));
		}
		return v;
	};
	{
		ofstream col_file(col_filename);
		cout << "writing file: " << col_filename << endl;

		if(!col_file.is_open()){
			cerr << "could not open: " << col_filename << endl;
			return 1;
		}

		// maximum degree of column is num_row
		for(int i=0;i<=num_rows;++i){
			col_file<<i<<separator<<gen_col()<<"\n";
		}
	}
	cout<<"done"<<endl;
	return 0;
}



