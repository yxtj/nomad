#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::vector;
using std::set;


int main(int argc, char** argv){
	int seed, dim;

	cout << "start" << endl;

	boost::program_options::options_description option_desc("abc");
	option_desc.add_options()
		("help,h", "produce help message")
		("seed", boost::program_options::value<int>(&seed)->default_value(12345),
			"RNG seed")
		("dim", boost::program_options::value<int>(&dim)->default_value(100),
			"dimension")
		;
	bool flag_help = false;

	try{

		boost::program_options::variables_map var_map;
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, option_desc), var_map);
		boost::program_options::notify(var_map);

		if(var_map.count("help")){
			flag_help = true;
		}

	} catch(std::exception& excep){
		cerr << "error: " << excep.what() << "\n";
		flag_help = true;
	} catch(...){
		cerr << "Exception of unknown type!\n";
		flag_help = true;
	}

	if(true == flag_help){
		cerr << option_desc << endl;
		return 1;
	}

	cout << "finish" << endl;

	return 0;

}
