import os
import subprocess

datasets = [("yahoo", 1.00, 0.0005, 0.05), ("netflix", 0.05, 0.008, 0.01), ("hugewiki", 0.01, 0.008, 0.005)]
numprocs = [32]
numcpus = [4]


for dataset, par_reg, lrate, drate in datasets:
    for numproc in numprocs:
        for numcpu in numcpus:

            task_name = "aws_bold_dsgd_%s_np%d_ncpu%d" % (dataset, numproc, numcpu)

            sub_fname = "../../TempFiles/%s.sub" % (task_name)
            log_fname = "../../TempFiles/%s.txt" % (task_name)

            ofile = open(sub_fname, "w")

            ofile.write("#!/bin/bash \n")
            ofile.write("#$ -cwd\n")
            ofile.write("#$ -pe orte %d \n" % (numproc))
            ofile.write("#$ -j y\n")
            ofile.write("#$ -e ../../TempFiles/%s.err \n" % task_name)
            ofile.write("#$ -o ../../TempFiles/%s.out \n" % task_name)

            #mpiexec -n 5 ./mpi-sgd-petsc rank lambda maxiter lrate decay_rate strategy nr_threads srcdir
            #strategy: 0 for fixed learning rate, 1 for bold driver
            ofile.write("\n mpirun ../../Code/competitor/mpi-sgd-omp 100 %f 300 %f %f 1 %d ../../Data/%s  > ../../Results/%s.txt\n" \
                % (par_reg, lrate, drate, numcpu, dataset, task_name))
            ofile.close()

            cmd = "qsub %s" % sub_fname
            print cmd
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
