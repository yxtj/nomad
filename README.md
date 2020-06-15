This project is based on the NOMAD project of Hyokun Yun [1]. It is modified/rewritten for my own research purpose. 



Information of the original version:

* NOMAD: Non-locking, stOchastic Multi-machine algorithm for Asynchronous and Decentralized matrix completion. NOMAD is a package for large-scale distributed matrix completion. Please refer to the paper [1] for detailed discussion on the algorithm.
* The original readme file is renamed as "README-old.md". The code is put into the branch named "version-of-Yun".
  * In order to compile the original code in modern compiler, you need to replace all `tbb::tick_count::interval_t` with `std::chrono::duration<double>` in the "nomad_body.hpp" file.

* NOMAD is under Apache License ver 2.0; see LICENSE file for detailed information.



Information of my version.

* The code is fully reorganized for readability. 

* Fix several bugs
* Add network controlling functions
* Add fault tolerance functions




Installation
============

Prerequisites
-------------

- MPI library, with multi-threading support
- Intel Thread Building Block (at least 4.1)
- CMake (at least 2.6)
- Boost library (at least 1.49)
- A C++ compiler which supports C++11



Compile
-----

To compile NOMAD, move to the root of this project and execute the `make` command. The executive files will be stored in the `bin/` folder.

Alternatively, you can move the the code directory of nomad `./Code/nomad` and use the `cmake` command to generate your own compiling folder.


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



[1] NOMAD: Non-locking, stOchastic Multi-machine algorithm for asynchronous and Decentralized Matrix Completion (Hyokun Yun, Hsiang-Fu Yu, Cho-Jui Hsieh, S.V.N. Vishwanathan, Inderjit Dhillon)