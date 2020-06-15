#ifndef _PMF_MPI_H_
#define _PMF_MPI_H_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <cmath>
#include <omp.h>

#include "smat.h"
#include "pmf.h"
#include <mpi.h>

typedef std::vector<double> vec_t;
typedef std::vector<vec_t> mat_t;

inline size_t get_procid() { 
	int mpi_rank(-1);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	assert(mpi_rank >= 0);
	return size_t(mpi_rank);
}

inline size_t nr_processors() {
	int mpi_size(-1);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	assert(mpi_size >= 0);
	return size_t(mpi_size);
}

#endif
