import os
import subprocess
import sys
datasets = [("yahoo", [0.25, 0.5, 1.00, 2, 4], 32), 
            ("netflix", [0.0125, 0.025, 0.05, 0.1, 0.2], 32), 
            ("hugewiki",  [0.0025, 0.005, 0.01, 0.02, 0.04], 64)]
numcpus = [4]
numrepeat = 1

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100

for dataset, par_regs, numproc in datasets:
    for par_reg in par_regs:
        for numcpu in numcpus:
        
            task_name = "%s_%s_%d_%d_%d_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg) 
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

            '''
            Usage:
            export MV2_ENABLE_AFFINITY=0
            mpiexec -n 4 mpi-ccdr1-omp [options] data_dir [model_filename]
            options:
            -s type : set type of solver (default 0)
            0 -- CCDR1 with fundec stopping condition
            -k rank : set the rank (default 10)
            -n threads : set the number of threads (default 4)
            -l lambda : set the regularization parameter lambda (default 0.1)
            -t max_iter: set the number of iterations (default 5)
            -T max_iter: set the number of inner iterations used in CCDR1 (default 5)
            -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)
            -p do_predict: do prediction or not (default 1)
            -q verbose: show information or not (default 0)
            -S do_random_suffle : conduct a random shuffle for columns (default 0)
            -N do_nmf: do nmf (default 0)
            '''

            # BUGBUG: import CCD++ MPI to NOMAD repository as well
            ofile.write("ibrun ../../../Code/competitor/mpi-ccd-r1-omp -k %d -l %f -t 30 -n %d ../../../Data/%s/ ../../../TempFiles/%s.model > %s \n" % (dim, par_reg, numcpu, dataset, task_name, log_fname))

            ofile.write("\n")
            ofile.close()

            qsub_command = "sbatch %s" % sub_fname

            p = subprocess.Popen(qsub_command, shell=True)
            p.communicate()
