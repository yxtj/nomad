#include "pool.h"

#include <algorithm>
#include <random>

using namespace std;

// ---- ColumnData ----

constexpr int ColumnData::SIGNAL_CP_START;
constexpr int ColumnData::SIGNAL_CP_CLEAR;
constexpr int ColumnData::SIGNAL_CP_LFINISH;
constexpr int ColumnData::SIGNAL_CP_RESUME;
constexpr int ColumnData::SIGNAL_LERROR;
constexpr int ColumnData::SIGNAL_TERMINATE;

char* ColumnData::serialize(char* cur_pos, const int dim){
	*(reinterpret_cast<int*>(cur_pos)) = col_index_;
	*(reinterpret_cast<long*>(cur_pos + sizeof(int))) = flag_;
	double* dest = reinterpret_cast<double*>(cur_pos + sizeof(long) + sizeof(int));
	std::copy(values_, values_ + dim, dest);
	return cur_pos;
}

void ColumnData::deserialize(char* cur_pos, const int dim){
	col_index_ = *(reinterpret_cast<int*>(cur_pos));
	flag_ = *(reinterpret_cast<long*>(cur_pos + sizeof(int)));
	double* dest = reinterpret_cast<double*>(cur_pos + sizeof(int) + sizeof(long));
	std::copy(dest, dest + dim, values_);
}

void ColumnData::set_perm(const int nthreads, std::mt19937_64& rng){
	pos_ = 0;
	for(int i = 0; i < nthreads; i++){
		perm_[i] = i;
	}
	std::shuffle(perm_, perm_ + nthreads, rng);
}

// ---- Pool ----

Pool::Pool(int dim, int num_threads) :
	queue_(), dim_(dim), num_threads_(num_threads), alloc_(),
	int_alloc_(), double_alloc_()
{
	
}

Pool::~Pool(){
	while(true){
		ColumnData* p_col = nullptr;
		bool succeed = queue_.try_pop(p_col);
		if(succeed){
			int_alloc_.deallocate(p_col->perm_, num_threads_);
			double_alloc_.deallocate(p_col->values_, dim_);
			alloc_.destroy(p_col);
			alloc_.deallocate(p_col, 1);
		} else{
			break;
		}
	}
}

ColumnData* Pool::allocate(){
	ColumnData* ret = alloc_.allocate(1);
	ret->error = 0.0;
	ret->perm_ = int_alloc_.allocate(num_threads_);
	ret->values_ = double_alloc_.allocate(dim_);
	return ret;
}

void Pool::push(ColumnData* p_col){
	p_col->error = 0.0;
	queue_.push(p_col);
}

ColumnData* Pool::pop(){
	ColumnData* ret = nullptr;
	bool succeed = queue_.try_pop(ret);

	if(succeed){
		return ret;
	} else{
		return allocate();
	}
}


