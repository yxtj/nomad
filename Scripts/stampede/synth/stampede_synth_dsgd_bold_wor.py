import os
import subprocess
import sys

numprocs = [4,16,32]
numcpus = [4]
numrepeat = 1

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100

par_reg = 0.01
lrate = 0.002
drate = 0.5

for numproc in numprocs:
    for numcpu in numcpus:

        task_name = "%s_%d_%d_%d_%f_%f_%f" % (exp_name, numproc, numcpu, dim, par_reg, lrate, drate) 
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
        ofile.write("#SBATCH -t 12:00:00 \n")
        ofile.write("#SBATCH -A NetworkMining \n")

        ofile.write("export MV2_ENABLE_AFFINITY=0 \n")
            
        log_fname = "../../../Results/%s.txt" % (task_name)

        #~/mpitest_final/mpi-sgd-omp
        #mpiexec -n 5 ./mpi-sgd-petsc rank lambda maxiter lrate decay_rate strategy nr_threads srcdir
        #strategy: 0 for fixed learning rate, 1 for bold driver
        ofile.write("ibrun ../../../Code/competitor/mpi-sgd-omp %d %f 500 %f %f 1 2 %d ../../../Data/synth/synth_%d > %s \n" % (dim, par_reg, lrate, drate, numcpu, numproc, log_fname))

        ofile.write("\n")
        ofile.close()

        qsub_command = "sbatch %s" % sub_fname

        p = subprocess.Popen(qsub_command, shell=True)
        p.communicate()
