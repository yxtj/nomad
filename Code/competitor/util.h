#ifndef MATUTIL
#define MATUTIL
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
#include <mpi.h>
#include <assert.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

using namespace std;
class rate_t;
class rateset_t; 
class blocks_t;
class RateComp;
class smat_t;
class testset_t;
typedef vector<double> vec_t;
typedef vector<vec_t> mat_t;
void load(const char *srcdir, blocks_t &blocks, testset_t &T);
void load(const char* srcdir, smat_t &R, testset_t &T);
void initial(mat_t &X, long n, long k);
void initial_col(mat_t &X, long k, long n);
void initial_col_ans(mat_t &X, long k, long n);
double dot(const vec_t &a, const vec_t &b);
double dot(const mat_t &W, const int i, const mat_t &H, const int j);
double dot(const mat_t &W, const int i, const vec_t &H_j);
double norm(const vec_t &a);
double norm(const mat_t &M);
double calloss(const smat_t &R, const mat_t &W, const mat_t &H);
double calobj(const smat_t &R, const mat_t &W, const mat_t &H, const double lambda, bool iscol=false);
double calobj(const blocks_t &blocks, const mat_t &W, const mat_t &H, const double lambda, bool iscol=false);

double calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol=false);
double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &H_t);
double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt);

class rate_t{
	public:
		int i, j; double v;
		rate_t(int ii=0, int jj=0, double vv=0): i(ii), j(jj), v(vv){}
};


// a block of rate set for PSGD
class rateset_t{
	public:
		const rate_t *from; long size_;
		rateset_t(const rate_t *from_=0, long size__=0): from(from_), size_(size__){}
		inline long size() const{return size_;}
		inline const rate_t& operator[](int i) const{return from[i];}
	
};

// block matrix format for PSGD
class blocks_t{
	public:
		blocks_t(){}
		blocks_t(int _B):B(_B),rows(0),cols(0),nnz(0){}
		void compressed_space(){
			vector<long>(block_ptr).swap(block_ptr);
			vector<rate_t>(allrates).swap(allrates);
			vector<unsigned>(nnz_row).swap(nnz_row);
			vector<unsigned>(nnz_col).swap(nnz_col);
			for(int bid = 1; bid <= B*B; ++bid)
				block_ptr[bid] += block_ptr[bid-1];
		}
		void from_mpi(long _rows, long _cols, long _nnz){
			rows =_rows,cols=_cols,nnz=_nnz;
			block_ptr.resize(B+1);
			allrates.resize(nnz);
			nnz_row.resize(rows);
			nnz_col.resize(cols);
			Bm = rows/B+((rows%B)?1:0); // block's row size 
			Bn = cols/B+((cols%B)?1:0); // block's col size
		}
		void load(long _rows, long _cols, long _nnz, const char *filename){ 
			rows =_rows,cols=_cols,nnz=_nnz;
			block_ptr = vector<long>(B*B+1,0);
			allrates = vector<rate_t>(nnz);
			nnz_row = vector<unsigned>(rows,0);
			nnz_col = vector<unsigned>(cols,0);
			Bm = rows/B+((rows%B)?1:0); // block's row size 
			Bn = cols/B+((cols%B)?1:0); // block's col size

			FILE *fp = fopen(filename,"r");
			for(long idx=0,i,j; idx < nnz; ++idx){
				double v;
				fscanf(fp,"%ld %ld %lf", &i, &j, &v);
				insert_rate(idx, rate_t(i-1,j-1,v)); // idx starts from 0
				++nnz_row[i-1];
				++nnz_col[j-1];
			}
			compressed_space(); // Need to call sort later.
			fclose(fp);
		}
		inline int bid_of_rate(int i, int j) const {return (i/Bm)*B + (j/Bn);}
		inline int size() const {return B*B;}
		inline rateset_t operator[] (int bid) const {
			return rateset_t(&allrates[block_ptr[bid]], block_ptr[bid+1]-block_ptr[bid]);
		}
		inline rateset_t getrateset(int bi, int bj) {
			return (*this)[bi * B + bj]; 
		}
		inline void insert_rate(long &idx, rate_t r) {
			allrates[idx] = r;
			block_ptr[(r.i/Bm)*B+r.j/Bn+1]++;
		}
		long nnz_of_row(int i) const {return nnz_row[i];}
		long nnz_of_col(int i) const {return nnz_col[i];}
		double get_global_mean(){
			double sum=0;
			for(int i=0; i < nnz; ++i)
				sum+=allrates[i].v;
			return sum/nnz;
		}
		void remove_bias(double bias=0){
			if(bias) 
				for(int i=0; i < nnz; ++i) allrates[i].v -= bias;
		}
		int B, Bm, Bn; long rows,cols,nnz;
		vector<rate_t> allrates;
		vector<long> block_ptr;
		vector<unsigned> nnz_row, nnz_col;
};

// Comparator for sorting rates into block format
class RateComp{
	public:
	const blocks_t *blocksptr;
	RateComp(const blocks_t *ptr): blocksptr(ptr){}
	bool operator()(rate_t x, rate_t y) const {
		return blocksptr->bid_of_rate(x.i, x.j) < blocksptr->bid_of_rate(y.i, y.j);
		/*
		// To get an unique ordering, use this.
		int bidx = blocksptr->bid_of_rate(x.i, x.j);
		int bidy = blocksptr->bid_of_rate(y.i, y.j);
		if(bidx!=bidy) return bidx < bidy;
		if(x.i != y.i) return x.i < y.i;
		return x.j < y.j;
		*/
	}

};

// Sparse matrix format CCS & RCS
// Access column fomat only when you use it..
class smat_t{
	public:
		long rows, cols;
		long nnz, max_row_nnz, max_col_nnz;
		double *val, *val_t;
		long *col_ptr, *row_ptr;
		long *col_nnz, *row_nnz;
		unsigned *row_idx, *col_idx;
		bool mem_alloc_by_me;
		smat_t():mem_alloc_by_me(false){ }
		smat_t(smat_t& m){ *this = m; mem_alloc_by_me = false;}

		
		void from_mpi(){
			mem_alloc_by_me=true;
			max_col_nnz = 0;
			for(long c=1; c<=cols; ++c) 
				max_col_nnz = max(max_col_nnz, col_ptr[c]-col_ptr[c-1]);
		}
		void print_mat(int host){
			for(int c = 0; c < cols; ++c) if(col_ptr[c+1]>col_ptr[c]){
				printf("%d: %ld at host %d\n", c, col_ptr[c+1]-col_ptr[c],host);
			}
		}
		void load(long _rows, long _cols, long _nnz, const char*filename) {
			rows =_rows,cols=_cols,nnz=_nnz;
			mem_alloc_by_me = true;
			val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
			row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);
			row_ptr = MALLOC(long, rows+1); col_ptr = MALLOC(long, cols+1);
			memset(row_ptr,0,sizeof(long)*(rows+1));
			memset(col_ptr,0,sizeof(long)*(cols+1));

			FILE *fp = fopen(filename, "r");
			// Read row-format data into CSR matrix
			for(long i=0,r,c; i<nnz; ++i){
				fscanf(fp,"%ld %ld %lf", &r, &c, &val_t[i]);
				row_ptr[r]++;
				col_ptr[c]++;
				col_idx[i] = c-1;
			}
			fclose(fp);
			// Calculate nnz for each row and col
			max_row_nnz = max_col_nnz = 0;
			for(long r=1; r<=rows; ++r) {
				max_row_nnz = max(max_row_nnz, row_ptr[r]);
				row_ptr[r] += row_ptr[r-1];
			}
			for(long c=1; c<=cols; ++c) {
				max_col_nnz = max(max_col_nnz, col_ptr[c]);
				col_ptr[c] += col_ptr[c-1];
			}
			// Transpose CSR into CSC matrix
			for(long r=0; r<rows; ++r){
				for(long i = row_ptr[r]; i < row_ptr[r+1]; ++i){
					long c = col_idx[i];
					row_idx[col_ptr[c]] = r; 
					val[col_ptr[c]++] = val_t[i];
				}
			}
			for(long c=cols; c>0; --c) col_ptr[c] = col_ptr[c-1];
			col_ptr[0] = 0;
		}
		long nnz_of_row(int i) const {return (row_ptr[i+1]-row_ptr[i]);}
		long nnz_of_col(int i) const {return (col_ptr[i+1]-col_ptr[i]);}
		double get_global_mean(){
			double sum=0;
			for(int i=0;i<nnz;++i) sum+=val[i];
			return sum/nnz;
		}
		void remove_bias(double bias=0){
			if(bias) {
				for(int i=0;i<nnz;++i) val[i]-=bias;
				for(int i=0;i<nnz;++i) val_t[i]-=bias;
			}
		}
		void free(void *ptr) {if(!ptr) ::free(ptr);}
		~smat_t(){
			if(mem_alloc_by_me) {
//				puts("Warnning: Somebody just free me.");
				if(val)free(val); if(val_t)free(val_t);
				if(row_ptr)free(row_ptr);if(row_idx)free(row_idx); 
				if(col_ptr)free(col_ptr);if(col_idx)free(col_idx);
			}
		}
		void clear_space() {
			if(val)free(val); if(val_t)free(val_t);
			if(row_ptr)free(row_ptr);if(row_idx)free(row_idx); 
			if(col_ptr)free(col_ptr);if(col_idx)free(col_idx);
			mem_alloc_by_me = false;
		}
		smat_t transpose(){
			smat_t mt;
			mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
			mt.val = val_t; mt.val_t = val;
			mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
			mt.col_idx = row_idx; mt.row_idx = col_idx;
			mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
			return mt;
		}
};


// Test set format
class testset_t{
	public:
	long rows, cols, nnz;
	vector<rate_t> T;
	testset_t(): rows(0), cols(0), nnz(0){}
	inline rate_t& operator[](const unsigned &idx) {return T[idx];}
	void load(long _rows, long _cols, long _nnz, const char *filename) {
		int r, c; 
		double v;
		rows = _rows; cols = _cols; nnz = _nnz;
		T = vector<rate_t>(nnz);
		FILE *fp = fopen(filename, "r");
		for(long idx = 0; idx < nnz; ++idx){
			fscanf(fp, "%d %d %lg", &r, &c, &v); 
			T[idx] = rate_t(r-1,c-1,v);
		}
		fclose(fp);
	}
	double get_global_mean(){
		double sum=0;
		for(int i=0; i<nnz; ++i) sum+=T[i].v;
		return sum/nnz;
	}
	void remove_bias(double bias=0){
		if(bias) for(int i=0; i<nnz; ++i) T[i].v-=bias;
	}
};

#endif
