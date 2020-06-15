import os
import subprocess
import sys

if len(sys.argv) < 2:
    print "usage: %s [block_num]" % sys.argv[0]
    exit(1)

datasets = [("netflix",0.05, 0.002), ("yahoo",1.00, 0.0001), ("hugewiki", 0.01, 0.001)]
numcpus = [30]
numproc = 1
numrepeat = 1

exp_name = sys.argv[0][:sys.argv[0].rfind('.py')]
dim = 100

count = 0
block_size = 4

block_num = int(sys.argv[1])

for numcpu in numcpus:
    for dataset, par_reg, lrate in datasets:

        current_block_num = count / block_size
        
        count += 1

        if current_block_num != block_num:
            continue

        task_name = "%s_%s_%d_%d_%d_%f_%f" % (exp_name, dataset, numproc, numcpu, dim, par_reg, lrate)  
        sub_fname = "../../../TempFiles/%s.sub" % (task_name)

        ofile = open(sub_fname, "w")
            
        ofile.write("#!/bin/sh -l \n")
        ofile.write("# FILENAME: %s \n" % (sub_fname));

        ofile.write("#SBATCH -J %s \n" % task_name)
        ofile.write("#SBATCH -e ../../../TempFiles/%s.err \n" % task_name)
        ofile.write("#SBATCH -o ../../../TempFiles/%s.out \n" % task_name)

        ofile.write("#SBATCH -n 1\n")
        ofile.write("#SBATCH -N 1\n")
        ofile.write("#SBATCH -p largemem \n")
        ofile.write("#SBATCH -t 12:00:00 \n")
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

        ofile.write("LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/gcc/4.7.1/lib\n")

        ofile.write("LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/gcc/4.7.1/lib64\n")

        ofile.write("GCC_LIB=/opt/apps/gcc/4.7.1/lib64\n")

        log_fname = "../../../Results/%s.txt" % (task_name)

        ofile.write("../../../Code/libmf-1.0/libmf train -t 300 --rand-shuffle --no-tr-rmse -k %d -p %f -q %f -g %f -ub -1 -ib -1 --no-use-avg -v ../../../Data/%s/te.bin -s %d ../../../Data/%s/tr.bin ../../../TempFiles/%s.model > %s\n" % (dim, par_reg, par_reg, lrate, dataset, numcpu, dataset, task_name, log_fname))

        #ofile.write("../../../Code/libmf-1.0/libmf train -t 300 --rand-shuffle --no-tr-rmse -k %d -p %f -q %f -g %f -ub -1 -ib -1 --no-use-avg -s %d ../../../Data/%s/tr.bin ../../../TempFiles/%s.model >> %s\n" % (dim, par_reg, par_reg, lrate, numcpu, dataset, task_name, log_fname))

        ofile.write("\n")
        ofile.close()

        qsub_command = "sbatch %s" % sub_fname
            
        p = subprocess.Popen(qsub_command, shell=True)
        p.communicate()
