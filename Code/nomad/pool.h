#ifndef POOL_H_
#define POOL_H_

//#include "nomad.hpp"

#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/tbb.h>
#include <random>

struct ColumnData{
public:
	// speical signals of "col_index_"
	// initialized once and used multiple times
	static constexpr int SIGNAL_CP_START = -1;
	static constexpr int SIGNAL_CP_CLEAR = -2;
	static constexpr int SIGNAL_CP_LFINISH = -3;
	static constexpr int SIGNAL_CP_RESUME = -4;
	static constexpr int SIGNAL_LERROR = -5;
	static constexpr int SIGNAL_TERMINATE = -6;

public:
	int col_index_; // its negative value is re-used as special signals
	long flag_;
	double error; // used for online termination check
	int source_; //set by receiver as MPI rank
	int *perm_;
	int pos_; // re-used as second parameter when col_index_ is negative
	double *values_;

	char* serialize(char* cur_pos, const int dim);

	void deserialize(char* cur_pos, const int dim);

	void set_perm(const int nthreads, std::mt19937_64& rng);

};

class Pool{

public:
	Pool(int dim, int num_threads);
	~Pool();

	ColumnData* allocate();

	void push(ColumnData* p_col);

	ColumnData* pop();

private:
	tbb::concurrent_queue<ColumnData *, tbb::cache_aligned_allocator<ColumnData *> > queue_;
	int dim_;
	int num_threads_;
	tbb::cache_aligned_allocator<ColumnData> alloc_;
	tbb::cache_aligned_allocator<int> int_alloc_;
	tbb::cache_aligned_allocator<double> double_alloc_;
};

#endif
