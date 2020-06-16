/*
 * Copyright (c) 2013 Hyokun Yun
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

#ifndef NOMAD_POOL_HPP_
#define NOMAD_POOL_HPP_

#include "nomad.hpp"
#include "tbb/tbb.h"
#include <algorithm>

namespace nomad {

struct ColumnData{
public:
	// speical signals of "col_index_"
	// initialized once and used multiple times
	static constexpr int SIGNAL_CP_START = -1;
	static constexpr int SIGNAL_CP_CLEAR = -2;
	static constexpr int SIGNAL_CP_LFINISH = -3;
	static constexpr int SIGNAL_CP_RESUME = -4;

public:
	int col_index_; // its negative value is re-used as special signals
	long flag_;
	int source_; //set by receiver as MPI rank
	int *perm_;
	int pos_; // re-used as second parameter when col_index_ is negative
	scalar *values_;

	char* serialize(char* cur_pos, const int dim){
		*(reinterpret_cast<int *>(cur_pos)) = col_index_;
		*(reinterpret_cast<long *>(cur_pos + sizeof(int))) = flag_;
		scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(long) + sizeof(int));
		std::copy(values_, values_ + dim, dest);
		return cur_pos;
	}

	void deserialize(char* cur_pos, const int dim){
		col_index_ = *(reinterpret_cast<int *>(cur_pos));
		flag_ = *(reinterpret_cast<long *>(cur_pos + sizeof(int)));
		scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(int)+ sizeof(long));
		std::copy(dest, dest + dim, values_);
	}

	void set_perm(const int nthreads, rng_type& rng){
		pos_=0;
		for(int i = 0; i < nthreads; i++){
			perm_[i] = i;
		}
		std::shuffle(perm_, perm_ + nthreads, rng);
	}

};

class Pool{

public:
	Pool(int dim, int num_threads, int init_size) :
			queue_(),dim_(dim),num_threads_(num_threads),alloc_(),
			int_alloc_(),scalar_alloc_()
	{
	}

	~Pool(){

		while(true){

			ColumnData *p_col = nullptr;
			bool succeed = queue_.try_pop(p_col);
			if(succeed){
				int_alloc_.deallocate(p_col->perm_, num_threads_);
				scalar_alloc_.deallocate(p_col->values_, dim_);
				alloc_.destroy(p_col);
				alloc_.deallocate(p_col, 1);
			}
			else{
				break;
			}

		}

	}

	ColumnData *allocate(){

		ColumnData *ret = alloc_.allocate(1);
		ret->perm_ = int_alloc_.allocate(num_threads_);
		ret->values_ = scalar_alloc_.allocate(dim_);
		return ret;

	}

	void push(ColumnData *p_col){
		queue_.push(p_col);
	}

	ColumnData *pop(){

		ColumnData *ret = nullptr;
		bool succeed = queue_.try_pop(ret);

		if(succeed){
			return ret;
		}
		else{
			return allocate();
		}

	}

private:
	tbb::concurrent_queue<ColumnData *, callocator<ColumnData *> > queue_;
	int dim_;
	int num_threads_;
	callocator<ColumnData> alloc_;
	callocator<int> int_alloc_;
	callocator<scalar> scalar_alloc_;
};

}

#endif
