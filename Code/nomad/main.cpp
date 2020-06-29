#include "nomad_body.h"
#include "nomad_option.h"
#include <glog/logging.h>

int main(int argc, char **argv) {
	//RealDataBody body;
	NomadOption option;
	if(!option.parse_command(argc, argv)){
		return 1;
	}

	//google::SetUsageMessage("invoke from mpirun.");
	//google::ParseCommandLineFlags(&argc, &argv, false);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	FLAGS_logtostderr = true;
	FLAGS_v = option.verbose_level;

	NomadBody body;
	return body.run(&option);

}
