import os
import subprocess
import sys

default_timeouts = [100, 200, 500, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 20000]
hugewiki_timeouts = [4000, 8000, 12000, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]

default_numprocs = [1,2,4,8,16,32]
hugewiki_numprocs = [4,8,16,32,64]
#hugewiki_numprocs = [1,2]

yahoo_regs = [0.25, 0.5, 1.00, 2, 4]
netflix_regs = [0.0125, 0.025, 0.05, 0.1, 0.2]
datasets = [("yahoo", 1.00, 0.0005, 0.05, default_timeouts, default_numprocs), 
            ("netflix", 0.05, 0.008, 0.01, default_timeouts, default_numprocs), 
            ("hugewiki", 0.01, 0.00066666, 0, hugewiki_timeouts, hugewiki_numprocs)]

numcpu = 4

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100

count = 0

for dataset, par_reg, lrate, drate, timeouts, numprocs in datasets:
    for numproc in numprocs:

        count += 1

        timeout_str = " ".join([str(int(float(timeout) / numproc / numcpu)) for timeout in timeouts])
        hours = 24 

        task_name = "%s_%s_%d_%d_%d_%f_%f_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg, lrate, drate)            
        sub_fname = "../../../TempFiles/%s.sub" % (task_name)

        ofile = open(sub_fname, "w")

        ofile.write("#!/bin/sh -l \n")
        ofile.write("# FILENAME: %s \n" % (sub_fname));

        ofile.write("#SBATCH -J %s \n" % task_name)
        ofile.write("#SBATCH -e ../../../TempFiles/%s.err \n" % task_name)
        ofile.write("#SBATCH -o ../../../TempFiles/%s.out \n" % task_name)

        ofile.write("#SBATCH -n %d\n" % numproc)
        ofile.write("#SBATCH -N %d\n" % numproc)
        ofile.write("#SBATCH -p normal \n")
        ofile.write("#SBATCH -t %d:00:00 \n" % hours)
        ofile.write("#SBATCH -A NetworkMining \n")

        ofile.write("source /etc/profile.d/tacc_modules.sh \n")
        ofile.write("export MV2_ENABLE_AFFINITY=0 \n")

        ofile.write("export PATH=/opt/apps/gcc/4.7.1/bin:$PATH\n")
        ofile.write("export PATH=/home1/01846/cjhsieh/newdir/cmake-2.8.10.2-Linux-i386/bin/:$PATH\n")
        ofile.write("module load intel/13.1.1.163\n")
        ofile.write("module load boost\n")
        ofile.write("source /opt/apps/intel/13/composer_xe_2013.3.163/tbb/bin/tbbvars.sh intel64\n")
        ofile.write("export BOOST_ROOT=$TACC_BOOST_DIR\n")
        ofile.write("export CXX=icc\n")

        ofile.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/gcc/4.7.1/lib\n")

        ofile.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/gcc/4.7.1/lib64\n")

        ofile.write("export GCC_LIB=/opt/apps/gcc/4.7.1/lib64\n")

        log_fname = "../../../Results/%s.txt" % (task_name)

        ofile.write("ibrun ../../../Code/nomad/nomad_double --timeout %s --path ../../../Data/%s --dim %d --reg %f --lrate %f --drate %f --nthreads %d > %s \n" % (timeout_str, dataset, dim, par_reg, lrate, drate, numcpu, log_fname) )

        ofile.write("\n")
        ofile.close()

        qsub_command = "sbatch %s" % sub_fname

        p = subprocess.Popen(qsub_command, shell=True)
        p.communicate()


