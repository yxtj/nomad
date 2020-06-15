import os
import subprocess
import sys

datasets = [("yahoo", 1.00, 0.0005, 0.5, 300), ("netflix", 0.05, 0.008, 0.5, 300), ("hugewiki", 0.01, 0.008, 0.5, 200)]
numprocs = [32]
numcpus = [4]

dim=100
exp_name=sys.argv[0][:sys.argv[0].rfind('.py')]

for dataset, par_reg, lrate, drate, num_iter in datasets:
    for numproc in numprocs:
        for numcpu in numcpus:

            task_name = "%s_%s_%d_%d_%d_%f_%f_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg, lrate, drate)

            sub_fname = "../../TempFiles/%s.sub" % (task_name)
            log_fname = "../../TempFiles/%s.txt" % (task_name)

            ofile = open(sub_fname, "w")

            ofile.write("#!/bin/bash \n")
            ofile.write("#$ -cwd\n")
            ofile.write("#$ -pe orte %d \n" % (numproc))
            ofile.write("#$ -j y\n")
            ofile.write("#$ -e ../../TempFiles/%s.err \n" % task_name)
            ofile.write("#$ -o ../../TempFiles/%s.out \n" % task_name)
            
            #export MV2_ENABLE_AFFINITY=0
            #mpiexec -n 4 ./mpi-sgd-omp rank lambda maxiter lrate decay_rate strategy stratum_strategy nr_threads srcdir
            #strategy: 0 for fixed learning rate, 1 for bold driver
            #stratum_strategy: 0 for semi-WOR, 1 for WR, 2 for WOR
            ofile.write("\n mpirun ../../Code/competitor/mpi-sgd-omp %d %f %d %f %f 1 2 %d ../../Data/%s  > ../../Results/%s.txt\n" \
                % (dim, par_reg, num_iter, lrate, drate, numcpu, dataset, task_name))
            ofile.close()

            cmd = "qsub %s" % sub_fname
            print cmd
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
