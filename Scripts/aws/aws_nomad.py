import sys
import os
import subprocess

default_timeouts = [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
hugewiki_timeouts = [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000, 30000, 40000, 50000, 70000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]

datasets = [("yahoo", 1.00, 0.0005, 0.05, default_timeouts), 
            ("netflix", 0.05, 0.008, 0.01, default_timeouts), 
            ("hugewiki", 0.01, 0.000666666, 0, hugewiki_timeouts)]
#datasets = [("netflix", 0.05, 0.008, 0.01, default_timeouts)] 
numprocs = [1,2,4,8,16,32]
#numprocs = [32]
numcpus = [2]

dim=100
exp_name=sys.argv[0][:sys.argv[0].rfind('.py')]

for dataset, par_reg, lrate, drate, timeouts in datasets:
    for numproc in numprocs:
        for numcpu in numcpus:

            task_name = "%s_%s_%d_%d_%d_%f_%f_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg, lrate, drate)
            timeout_str = " ".join([str(timeout/numcpu/numproc) for timeout in timeouts])

            sub_fname = "../../TempFiles/%s.sub" % (task_name)
            log_fname = "../../TempFiles/%s.txt" % (task_name)

            ofile = open(sub_fname, "w")

            ofile.write("#!/bin/bash \n")
            ofile.write("#$ -cwd\n")
            ofile.write("#$ -pe orte %d \n" % (numproc))
            ofile.write("#$ -j y\n")
            ofile.write("#$ -e ../../TempFiles/%s.err \n" % task_name)
            ofile.write("#$ -o ../../TempFiles/%s.out \n" % task_name)

            ofile.write("\nmpirun ../../Code/nomad/nomad_double --timeout %s --dim %d --reg %f --lrate %f --drate %f --nthreads %d --path ../../Data/%s > ../../Results/%s.txt\n" \
                % (timeout_str, dim, par_reg, lrate, drate, numcpu, dataset, task_name))
            ofile.close()

            cmd = "qsub %s" % sub_fname
            print cmd
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
