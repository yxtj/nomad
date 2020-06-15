import os
import subprocess
import sys

datasets = [("yahoo", 1.00, 30), ("netflix", 0.05, 30), ("hugewiki", 0.01, 30)]
numprocs = [32]
numcpus = [4]

dim=100
exp_name=sys.argv[0][:sys.argv[0].rfind('.py')]

for dataset, par_reg, num_iter in datasets:
    for numproc in numprocs:
        for numcpu in numcpus:

            task_name = "%s_%s_%d_%d_%d_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg)

            sub_fname = "../../TempFiles/%s.sub" % (task_name)
            log_fname = "../../TempFiles/%s.txt" % (task_name)

            ofile = open(sub_fname, "w")

            ofile.write("#!/bin/bash \n")
            ofile.write("#$ -cwd\n")
            ofile.write("#$ -pe orte %d \n" % (numproc))
            ofile.write("#$ -j y\n")
            ofile.write("#$ -o ../../TempFiles/%s.out \n" % task_name)
            ofile.write("#$ -e ../../TempFiles/%s.err \n" % task_name)

            #root@master:/vishydata/nomad/Code/competitor# ./mpi-ccd-r1-omp
            #Usage:
            #export MV2_ENABLE_AFFINITY=0
            #mpiexec -n 4 mpi-ccdr1-omp [options] data_dir [model_filename]
            #options:
            #-s type : set type of solver (default 0)
            #0 -- CCDR1 with fundec stopping condition
            #-k rank : set the rank (default 10)
            #-n threads : set the number of threads (default 4)
            #-l lambda : set the regularization parameter lambda (default 0.1)
            #-t max_iter: set the number of iterations (default 5)
            #-T max_iter: set the number of inner iterations used in CCDR1 (default 5)
            #-e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)
            #-p do_predict: do prediction or not (default 1)
            #-q verbose: show information or not (default 0)
            #-S do_random_suffle : conduct a random shuffle for columns (default 0)
            #-N do_nmf: do nmf (default 0)
            # mpirun -np 2 -hosts master,master ./mpi-ccd-r1-omp -k 100 -n 2 -l 0.05 ../../Data/netflix
            ofile.write("\n mpirun ../../Code/competitor/mpi-ccd-r1-omp -k %d -l %f -t %d -n %d ../../Data/%s  > ../../Results/%s.txt\n" \
                % (dim, par_reg, num_iter, numcpu, dataset, task_name))
            ofile.close()

            cmd = "qsub %s" % sub_fname
            print cmd
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
