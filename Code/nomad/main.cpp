//#include "realdata_body2.hpp"
#include "nomad_body.h"
#include "nomad_option.h"

int main(int argc, char **argv) {

	//RealDataBody body;
	NomadOption option;
	if(!option.parse_command(argc, argv)){
		return 1;
	}
	NomadBody body;
	return body.run(&option);

}
