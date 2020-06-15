NOMAD: Non-locking, stOchastic Multi-machine algorithm 
       for Asynchronous and Decentralized matrix completion

NOMAD is a package for large-scale distributed matrix completion.
Please refer to the paper [1] for detailed discussion on the algorithm.


Table of Contents
=================

- License
- Installation
  - Prerequisites
  - NOMAD
  - Competitors
- Usage
  - Data processing
  - Execution
- Scripts for reproducing results 


License
=======

NOMAD is under Apache License ver 2.0; see LICENSE file for detailed information.

However, LibMF, distributed together as a competitor to NOMAD, is not under the same license.
Please refer to COPYRIGHT file in ./Code/libmf-1.0 about its license policy.


Installation
============

Prerequisites
-------------

Currently, NOMAD only supports UNIX-based systems.  NOMAD is dependent on following libraries:

- MPI library, with multi-threading support (MPICH2 and MVAPICH2 are recommended)
- Intel Thread Building Block (at least 4.1)
- CMake (at least 2.6)
- Boost library (at least 1.49)

Also, to compile NOMAD a modern C++ compiler which supports C++11 is needed.
We recommend g++ ver. 4.7.1 or higher, or Intel C++ compiler ver. 13.1.1 or higher.



NOMAD
-----

To compile NOMAD, move to the path where the source code of NOMAD is located:

$ cd ./Code/nomad

Then, run CMake to generate Makefiles:

$ cmake .

If CMake was succeed, you can run 'make' to compile NOMAD

$ make


Competitors
-----------

  a. DGSD, DSGD++ and CCD++

To compile DSGD, DSGD++ and CDD++, the procedure is similar as above.
First, change the current directory to the location where competitor source code is located:

$ cd ./Code/competitor

The remaining procedure is the same:

$ cmake .
$ make


  b. FPSGD**

LibMF, implementation of FPSGD**, is located in ./Code/libmf-1.0.
Please refer to COPYRIGHT and README file in that directory for license and
installation instructions.


Usage
-----

  a. Data Processing

To use NOMAD, you need to convert a text data file to a binary format NOMAD can read.
The original text data for training set should be in tab-delimited form, as follows:

$ cat ./Data/tutorial/train.txt

user_A     item_1   2.0
user_A     item_2   3.0
user_B	   item_1   4.0
user_B	   item_3   7.0

...

Test dataset should be prepared accordingly.  Then, you can execute the conversion script:

$ python ./Scripts/convert.py ./Data/tutorial/train.txt ./Data/tutorial/test.txt ./Data/tutorial/

This will generate 'train.dat' and 'test.dat' on the destination directory './Data/tutorial/'.


Execution
---------

NOMAD has two executables, 'nomad_float' and 'nomad_double'.
The former uses single-precision, while the latter uses double-precision.

You can execute nomad with --help command to see the list of options

$ ./nomad_double --help
nomad options:
  -h [ --help ]                         produce help message
  --nthreads arg (=4)                   number of threads to use (0: automatic)
  -l [ --lrate ] arg (=0.001)           learning rate
  -d [ --drate ] arg (=0.10000000000000001)
                                        decay rate
  -r [ --reg ] arg (=1)                 regularization parameter lambda
  -s [ --seed ] arg (=12345)            seed value of random number generator
  -t [ --timeout ] arg (=10.0)          timeout seconds until completion
  -p [ --ptoken ] arg (=1024)           number of tokens in the pipeline
  -d [ --dim ] arg (=100)               dimension of latent space
  --reuse arg (=1)                      number of column reuse
  --pause arg (=1)                      number of column reuse
  --r0delay arg (=1)                    arbitrary network delay added to
                                        communication of rank 0 machine
  --output arg                          path of the file the result will be
                                        printed into
  --path arg 
                                        path of data


Scripts for reproducing results 
===============================

Results in the paper [1] can be reproduced by executing scripts
in the following locations:

4.2. Scaling in Number of Cores: ./Scripts/stampede/single
4.3. Scaling as Fixed Dataset is Distributed Across Processors:
     ./Scripts/stampede/multi
4.4. Scaling on Commodity Hardware:
     ./Scripts/aws/
4.5. Scaling as both Dataset Size and Number of Machines grows:
     ./Scripts/stampede/synth


To execute scripts, following directories should contain needed data:

./Data/netflix: Netflix competition dataset
./Data/yahoo: Yahoo! Music dataset
./Data/hugewiki: Hugewiki dataset

Results will be written as text files in ./Results



[1] NOMAD: Non-locking, stOchastic Multi-machine algorithm
    for asynchronous and Decentralized Matrix Completion
    (Hyokun Yun, Hsiang-Fu Yu, Cho-Jui Hsieh, 
    S.V.N. Vishwanathan, Inderjit Dhillon)