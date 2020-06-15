import os
import subprocess
import sys

datasets = [("yahoo", 1.00, 0.0005, 0.5, 32), 
            ("netflix", 0.05, 0.008, 0.5, 32), 
            ("hugewiki", 0.01, 0.008, 0.5, 64)]
numcpus = [4]
numrepeat = 1

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100

for dataset, par_reg, lrate, drate, numproc in datasets:
    for numcpu in numcpus:
        
        task_name = "%s_%s_%d_%d_%d_%f_%f_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg, lrate, drate) 
        sub_fname = "../../../TempFiles/%s.sub" % (task_name)

        ofile = open(sub_fname, "w")
            
        ofile.write("#!/bin/sh -l \n")
        ofile.write("# FILENAME: %s \n" % (sub_fname));

        ofile.write("#SBATCH -J %s \n" % task_name)
        ofile.write("#SBATCH -e ../../../TempFiles/%s.err \n" % task_name)
        ofile.write("#SBATCH -o ../../../TempFiles/%s.out \n" % task_name)

        ofile.write("#SBATCH -n %d\n" % (numproc))
        ofile.write("#SBATCH -N %d\n" % (numproc))
        ofile.write("#SBATCH -p normal \n")
        ofile.write("#SBATCH -t 4:00:00 \n")
        ofile.write("#SBATCH -A NetworkMining \n")

        ofile.write("export MV2_ENABLE_AFFINITY=0 \n")

        log_fname = "../../../Results/%s.txt" % (task_name)

        #export MV2_ENABLE_AFFINITY=0
        #mpiexec -n 4 ./mpi-dsgdpp rank lambda maxiter lrate decay_rate strategy stratum_strategy nr_threads srcdir
        #strategy: 0 for fixed learning rate, 1 for bold driver
        #stratum_strategy: 0 for semi-WOR, 1 for WR, 2 for WOR
        ofile.write("ibrun ../../../Code/competitor/mpi-dsgdpp %d %f 500 %f %f 1 2 %d ../../../Data/%s/ > %s \n" % (dim, par_reg, lrate, drate, numcpu, dataset, log_fname))

        ofile.write("\n")
        ofile.close()

        qsub_command = "sbatch %s" % sub_fname

        p = subprocess.Popen(qsub_command, shell=True)
        p.communicate()
