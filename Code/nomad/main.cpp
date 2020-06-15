//#include "realdata_body2.hpp"
#include "nomad_body.h"
#include "nomad_option.h"

int main(int argc, char **argv) {

	//RealDataBody body;
	NomadOption option;
	option.parse_command(argc, argv);
	if(!option.is_option_OK()){
		return 1;
	}
	NomadBody body;
	return body.run(&option);

}
