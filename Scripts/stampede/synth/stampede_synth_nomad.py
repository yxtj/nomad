import os
import subprocess
import sys

timeouts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000]
numprocs = [4,16,32]
numcpu = 4
par_reg = 0.01
lrate = 0.0008
drate = 0.0001

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100


for numproc in numprocs:

    timeout_str = " ".join([str(timeout) for timeout in timeouts])

    hours = 9 #int(float(real_timeout) / 3600 + 5)

    task_name = "%s_synth_%d_%d_%d_%f_%f_%f" % (exp_name, numproc, numcpu, dim, par_reg, lrate, drate)            
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

    ofile.write("ibrun ../../../Code/nomad/nomad_double --timeout %s --path ../../../Data/synth/synth_%d --dim %d --reg %f --lrate %f --drate %f --nthreads %d > %s \n" % (timeout_str, numproc, dim, par_reg, lrate, drate, numcpu, log_fname) )

    ofile.write("\n")
    ofile.close()

    qsub_command = "sbatch %s" % sub_fname

    p = subprocess.Popen(qsub_command, shell=True)
    p.communicate()
