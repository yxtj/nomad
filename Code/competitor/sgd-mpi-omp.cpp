//#include "util.h"
#include "util-mpi.h"
#include <iostream>
#include <iterator>

#define kind dynamic

#include <random>
typedef std::mt19937_64 rng_type;

#define SEED_VALUE 12345

//global variables for MPI
vector<int> block_row_ptr, block_col_ptr, block_row_cnt, block_col_cnt;
vector<long> block_row_ptr_k, block_col_ptr_k, block_row_cnt_k, block_col_cnt_k;

enum { FIXED=0, BOLDDRIVER=1 };

enum {SEMIWOR_STRATUM=0, WR_STRATUM=1, WOR_STRATUM=2};
enum {ST_FREE, ST_DONE};

inline size_t getprocid() { 
	int mpi_rank(-1);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	assert(mpi_rank >= 0);
	return size_t(mpi_rank);
}
inline size_t size() {
	int mpi_size(-1);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	assert(mpi_size >= 0);
	return size_t(mpi_size);
}

class mpi_blocks_t{
	public:
		mpi_blocks_t(){}
		//mpi_blocks_t(int _B, int _innerB): B(_B),innerB(_innerB){ }
		void set_parameters(int _B, int _innerB, int _rows, int _cols, long _global_nnz, int _startrow) {
			B = _B; innerB = _innerB;
			rows = _rows; cols = _cols; 
			global_nnz = _global_nnz; startrow = _startrow;
			Bm = rows/B+((rows%B)?1:0); // block's row size 
			Bn = cols/B+((cols%B)?1:0); // block's col size
			innerBm = Bm/innerB+((Bm%innerB)?1:0); // innerblock's row size
			innerBn = Bn/innerB+((Bn%innerB)?1:0); // innerblock's col size
			nnz_row.resize(rows);
			nnz_col.resize(cols);
			block_ptr.resize(B*innerB*innerB+1);
		}
		void compressed_space(){
			vector<long>(block_ptr).swap(block_ptr);
			vector<rate_t>(allrates).swap(allrates);
			vector<unsigned>(nnz_row).swap(nnz_row);
			vector<unsigned>(nnz_col).swap(nnz_col);
			for(int bid = 1; bid <= B*innerB*innerB; ++bid)
				block_ptr[bid] += block_ptr[bid-1];
		}
		inline int global_row(int localr) const {return startrow+localr;}
		inline int local_row(int r) const {return r-startrow;}
		inline int bid_of_rate(int r, int c) const {
			int out_bid = c/Bn; // outer block id
			int bi = (r-startrow)/innerBm; // (i, j) index for inner blocks
			int bj = (c-out_bid*Bn)/innerBn; 
			return out_bid*innerB*innerB + bi*innerB + bj;
		}

		inline int size() const {return B*innerB*innerB;}
		inline rateset_t operator[] (int bid) const {
			return rateset_t(&allrates[block_ptr[bid]], block_ptr[bid+1]-block_ptr[bid]);
		}
		inline rateset_t  getrateset(int out_bid, int bi, int bj) {
			return (*this)[out_bid*innerB*innerB + bi*innerB + bj];
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
		int B, Bm, Bn, innerB, innerBm, innerBn;
		int rows,cols,startrow;
		long nnz;
		long global_nnz;
		vector<rate_t> allrates;
		vector<long> block_ptr;
		vector<unsigned> nnz_row, nnz_col;
};

#define ROOT 0

void Two2One(mat_t &W, vec_t &W1, int k, int begin, int end) {
	int count = 0;
	for ( int i=begin ; i<end ; i++)
		for ( int t=0 ; t<k ; t++ )
			W1[count++] = W[i][t];
}

void One2Two(vec_t &W1, mat_t &W, int k, int begin, int end) {
	int count = 0; 
	for ( int i=begin ; i<end ; i++ )
		for (int t=0 ; t<k ; t++ )
			W[i][t] = W1[count++];
}

// Before this function call, make sure
// 1. H is updated for each process 
// 2. W can be partially stored in each process.
double calobj_mpi(const mpi_blocks_t &blocks, const mat_t &W, const mat_t &H, const double lambda, double *transtime=NULL, double *computetime=NULL){
	int procid = getprocid();
	double loss = 0, totalloss;
	double reg = 0, totalreg;
	double obj = 0;
	double tmpstart = omp_get_wtime();
#pragma omp parallel for schedule(kind) reduction(+:loss)
	for(size_t idx = 0; idx < blocks.allrates.size(); idx++) {
		const rate_t &r = blocks.allrates[idx];
		double diff = -r.v;
		diff += dot(W[r.i], H[r.j]);
		loss += diff*diff;
	}
	/*
	   for(int id = blocks.B*blocks.innerB*blocks.innerB-1; id >=0; --id){
	   const rateset_t &rateset = blocks[id];
	   for(int idx = rateset.size()-1; idx >=0; --idx){
	   double diff = -rateset[idx].v;
	   diff += dot(W[rateset[idx].i], H[rateset[idx].j]);
	   loss += diff*diff;
	   }
	   }
	   */
	int begin = block_row_ptr[procid], end = block_row_ptr[procid+1];
	reg = 0;
#pragma omp parallel for schedule(kind) reduction(+:reg)
	for(int r = begin; r < end; ++r) 
		reg += blocks.nnz_of_row(r)*norm(W[r]);
	begin = block_col_ptr[procid], end = block_col_ptr[procid+1];
#pragma omp parallel for schedule(kind) reduction(+:reg)
	for(int c = begin; c < end; ++c) 
		reg += blocks.nnz_of_col(c)*norm(H[c]);
	if(computetime) *computetime += omp_get_wtime() - tmpstart;

	tmpstart = omp_get_wtime();
	MPI_Reduce(&loss, &totalloss, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
	MPI_Reduce(&reg, &totalreg, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
	if(procid==ROOT) obj = totalloss+lambda*totalreg;
	MPI_Bcast(&obj, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	if(transtime) *transtime += omp_get_wtime()-tmpstart;

	return obj;
}

// Before this function call, make sure
// 1. H is updated for each process 
// 2. W can be partially stored in each process.
double calrmse_mpi(const mpi_blocks_t &blocks, const mat_t &W, const mat_t &H){
	int procid = getprocid();
	double loss = 0, totalloss = 0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
	for(int id = blocks.B*blocks.innerB*blocks.innerB-1; id >=0; --id){
		const rateset_t &rateset = blocks[id];
		for(long idx = rateset.size()-1; idx >=0; --idx){
			double diff = -rateset[idx].v;
			diff += dot(W[rateset[idx].i], H[rateset[idx].j]);
			loss += diff*diff;
		}
	}
	double tmpstart = omp_get_wtime();
	MPI_Reduce(&loss, &totalloss, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
	return procid==ROOT? sqrt(totalloss/blocks.global_nnz): 0;
}

void sgd(const rateset_t &rateset, mat_t &W, mat_t &H, 
		int k, double lambda, double lrate, unsigned *seed, int maxiter=1){
	long blocknnz = rateset.size();
	if (blocknnz == 0) return;
	while(maxiter--){
		long cnt = blocknnz;
		if(cnt < 0) {
			fprintf(stderr, "procid %ld: overflow happens nnz %ld cnt %ld\n", getprocid(), blocknnz, cnt);
			fflush(stdout);
		}
		while(cnt--){
			//long idx = rand_r(seed) % blocknnz;
			//long idx = rand() % blocknnz;
			long idx = cnt % blocknnz;
			int i = rateset[idx].i, j = rateset[idx].j;
			vec_t &Wi = W[i], &Hj = H[j];
			double err = dot(Wi,Hj)-rateset[idx].v;
			for(int kk = 0; kk < k; ++kk){
				double tmp = Wi[kk];
				Wi[kk] -= lrate*(err*Hj[kk]+lambda*Wi[kk]);
				Hj[kk] -= lrate*(err*tmp+lambda*Hj[kk]);
			}
		}
	}
}

void print_perm(const vector<vector<int> >& perm_matrix){
	for(int i=0; i<perm_matrix.size(); i++){
		cout << "Row: " << i << ": ";
		copy(perm_matrix[i].begin(), perm_matrix[i].end(), std::ostream_iterator<int>(cout, "  "));
		cout << endl;
	}
	cout << endl << endl;
}
void dsgd_mpi(const mpi_blocks_t &blocks, const mpi_blocks_t &testblocks, mat_t &W, mat_t &H, 
		int k, double lambda, double lrate, double decay_rate, int strategy, int stratum_strategy, testset_t &T, int maxiter=10){
	long B = blocks.B, m = blocks.rows, n = blocks.cols;
	long innerB = blocks.innerB;
	long nnz = blocks.nnz;
	int procid = getprocid();
	double obj, oldobj;

	// Time spent in iterations. Includes both communication and computation time
	double totaltime=0;
	double totalstart=0;

	// Time spent in SGD iterations   
	double computetime=0;
	double computestart=0;

	// Time spent in computing objective function value.
	double objtime=0;
	double objstart=0;

	// Time spent in communication 
	double commtime=0;
	double commstart=0;

	double cur_lrate = lrate;
	int sync_signal;
	vec_t localW1, localH1; // for slave node
	vec_t W1, H1;
	H1.resize(n*k);
	localH1.resize(blocks.Bn*k);

	MPI_Status stat;

	vector<vector<int> > perm_matrix(B, vector<int>(B,0));
	vector<int> indices(B, 0);
	vector<int> current_perm(B,0);
	vector<int> prev_perm(B,0);
	vector<int> tmp_buffer(B,0);

	vector<int> inner_cur_perm(innerB);
	vector<int> inner_perm(innerB);
	vector<int> inner_offset(innerB);

	vector<int> block_to_offset(B+1);
	vector<int> block_to_cnt(B+1);

	for(int i=0; i<B; i++){
		indices[i]=i;
		current_perm[i]=i;
		for(int j=0; j<B; j++){
			perm_matrix[i][j]=(i+j)%B;
		}
	}

	for(int i=0; i<innerB; i++)
		inner_perm[i] = inner_offset[i] = inner_cur_perm[i] = i;

	// Time to calculate the initial obj is not counted 
	if (strategy == BOLDDRIVER) {
		// obj is required for bold-driver
		objstart = omp_get_wtime();
		oldobj = calobj_mpi(blocks, W, H, lambda);
		objtime += omp_get_wtime() - objstart;
		// Add the initial objective function calculation time to the total time for bold driver 
		totaltime += objtime;
	}

	const unsigned seed = 34586;
	for(int iter = 1; iter <= maxiter; ++iter){

		MPI_Barrier(MPI_COMM_WORLD);
		totalstart = omp_get_wtime();

		srand(seed+iter);  
		// Shuffle the rows 
		random_shuffle(perm_matrix.begin(), perm_matrix.end());

		// Now shuffle the columns 
		// First shuffle the indices of the columns 
		random_shuffle(indices.begin(), indices.end());

		// Now actually shuffle the entries of perm_matrix
		for(int i=0; i<B; i++){
			copy(perm_matrix[i].begin(), perm_matrix[i].end(), tmp_buffer.begin());
			for(int j=0; j<B; j++)
				perm_matrix[i][indices[j]]=tmp_buffer[j];
		}

		if (strategy == FIXED)
			cur_lrate = 1.5*lrate/(1.0+decay_rate*pow(iter, 1.5));

		for(int s = 0; s < B; ++s) {
			copy(current_perm.begin(), current_perm.end(), prev_perm.begin());
			copy(perm_matrix[s].begin(), perm_matrix[s].end(), current_perm.begin()); 

			// Find which block you should own for this iteration
			// and which block you owned in the previous iteration
			int new_block=current_perm[procid];
			int old_block=prev_perm[procid];

			// Find whom to receive new_block from 
			// and whom to send old_block to
			int recv_id=-1;
			int send_id=-1;
			for(int i=0; i<B; i++){
				if(prev_perm[i]==new_block)
					recv_id=i;
				if(current_perm[i]==old_block)
					send_id=i;
			}
			assert(recv_id!=-1);
			assert(send_id!=-1);

			// Transfer the old block of H and receive the new block
			commstart = omp_get_wtime();
			int old_begin = block_col_ptr[old_block];
			int old_end = block_col_ptr[old_block]+block_col_cnt[old_block];
			Two2One(H,localH1,k,old_begin,old_end);
			MPI_Sendrecv_replace(&localH1[0],blocks.Bn*k,MPI_DOUBLE,send_id,1,recv_id, 1, MPI_COMM_WORLD, &stat);

			// Copy from the 1-d buffer to the 2-d buffer
			int new_begin = block_col_ptr[new_block];
			int new_end = block_col_ptr[new_block]+block_col_cnt[new_block];
			One2Two(localH1,H,k,new_begin,new_end);
			commtime += omp_get_wtime() - commstart;

			computestart = omp_get_wtime(); 
			// SGD computation
			//
			// random stratum for inner blocks
			if(stratum_strategy == WOR_STRATUM) {
				std::random_shuffle(inner_perm.begin(), inner_perm.end());
				std::random_shuffle(inner_offset.begin(), inner_offset.end());
			}
			for(int i = 0; i < innerB; ++i) {
				for(int ii = 0; ii < innerB; ii++)
					inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(kind) 
				for(int bi = 0; bi< innerB; bi++) {
					unsigned seed = bi * iter;
					//int bj = (bi+i)%innerB;
					int bj = inner_cur_perm[bi];
					int bid = new_block*innerB*innerB + bi*innerB + bj;
					sgd(blocks[bid], W, H, k, lambda, cur_lrate, &seed, 1);
				}
			}
			computetime += omp_get_wtime() - computestart;
		} // inner loop ends here

		if(strategy != BOLDDRIVER) {
			MPI_Barrier(MPI_COMM_WORLD);
			totaltime += omp_get_wtime() - totalstart;
		}

		// Allgather H
		commstart = omp_get_wtime();
		for(int ss = 0; ss < B; ss++) {
			block_to_offset[ss] = block_col_ptr_k[current_perm[ss]];
			block_to_cnt[ss] = block_col_cnt_k[current_perm[ss]];
		}
		MPI_Allgatherv(&localH1[0],block_to_cnt[procid],MPI_DOUBLE, &H1[0], &block_to_cnt[0], &block_to_offset[0],MPI_DOUBLE, MPI_COMM_WORLD);
		One2Two(H1,H,k,0,n);
		commtime += omp_get_wtime() - commstart;

		// BOLDDRIVER should include the obj calculation time
		if(strategy == BOLDDRIVER) {
			// Calculate obj to update learning rate for bold driver 
			objstart = omp_get_wtime();
			obj = calobj_mpi(blocks, W, H, lambda);
			if(obj > oldobj) cur_lrate *= decay_rate; else cur_lrate *= 1.05;
			oldobj = obj;
			objtime += omp_get_wtime() - objstart;

			MPI_Barrier(MPI_COMM_WORLD);
			totaltime += omp_get_wtime() - totalstart;
		}

		// Timing calculation
		double avg_totaltime = 0, avg_computetime = 0, max_computetime = 0, min_computetime = 1e200;
		MPI_Reduce(&totaltime, &avg_totaltime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &avg_computetime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &max_computetime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &min_computetime, 1, MPI_DOUBLE, MPI_MIN, ROOT, MPI_COMM_WORLD);
		avg_totaltime /= blocks.B; avg_computetime /= blocks.B;

		if(procid==ROOT){
			printf("iter %d totaltime %.6g computetime %.6g max %.6g min %.6g idle %.2g%% ", iter, 
					avg_totaltime, avg_computetime, max_computetime, min_computetime, 100.0*(avg_totaltime-avg_computetime)/(avg_totaltime));
			if(strategy == BOLDDRIVER) 
				printf("obj %.6g ", obj);
		}

		fflush(stdout);

		// RMSE calculation
		double rmse = calrmse_mpi(testblocks, W, H);
		if(procid==ROOT){
			if(testblocks.global_nnz!=0){
				printf("rmse %.6g", rmse);
			}
			puts("");
			fflush(stdout);
		} 
	}
}

void gen_block2(int k, int numprocs) {
	block_row_ptr_k.resize(numprocs+1);
	block_row_cnt_k.resize(numprocs);
	block_col_ptr_k.resize(numprocs+1);
	block_col_cnt_k.resize(numprocs);
	for ( int i=0 ; i<numprocs+1 ; i++ )
	{
		block_row_ptr_k[i] = (long)block_row_ptr[i]*k;
		block_col_ptr_k[i] = (long)block_col_ptr[i]*k;
	}
	for ( int i=0 ; i<numprocs ; i++ )
	{
		block_row_cnt_k[i] = (long)block_row_cnt[i]*k;
		block_col_cnt_k[i] = (long)block_col_cnt[i]*k;
	}
}

void gen_block_ptr(int m, int n, int numprocs, int nr_thread) {

  // make data division same to NOMAD
  int num_parts = numprocs * nr_thread;

  int blocksize;
  // blocksize = m/numprocs + ((m%numprocs)?1:0); 
  blocksize = (m/num_parts + ((m%num_parts)?1:0)) * nr_thread;
  block_row_ptr.resize(numprocs+1);
  block_row_cnt.resize(numprocs,blocksize);
  block_row_cnt[numprocs-1] = m-blocksize*(numprocs-1);
  block_row_ptr[0] = 0;
  for(int i = 1; i <= numprocs; ++i)
    block_row_ptr[i] = block_row_ptr[i-1]+block_row_cnt[i-1];

  blocksize = n/numprocs + ((n%numprocs)?1:0); 
  block_col_ptr.resize(numprocs+1);
  block_col_cnt.resize(numprocs,blocksize);
  block_col_cnt[numprocs-1] = n-blocksize*(numprocs-1);
  block_col_ptr[0] = 0;
  for(int i = 1; i <= numprocs; ++i)
    block_col_ptr[i] = block_col_ptr[i-1]+block_col_cnt[i-1];
  
  return;
}

void scatter_blocks(blocks_t &blocks, int numprocs) {
	int B = blocks.B;
	for(int pid = 1; pid <= numprocs; ++pid) {
		long start = blocks.block_ptr[(pid-1)*B];
		long end = blocks.block_ptr[pid*B];
		long nnz = end - start; 
		MPI_Send(&nnz,1,MPI_LONG,pid,9,MPI_COMM_WORLD);
		MPI_Send(&blocks.allrates[start],sizeof(rate_t)*nnz,MPI_CHAR,pid,3,MPI_COMM_WORLD);
		MPI_Send(&blocks.block_ptr[(pid-1)*B],B+1,MPI_LONG,pid,9,MPI_COMM_WORLD);
		MPI_Send(&blocks.nnz_row[0],blocks.rows,MPI_UNSIGNED,pid,9,MPI_COMM_WORLD);
		MPI_Send(&blocks.nnz_col[0],blocks.cols,MPI_UNSIGNED,pid,9,MPI_COMM_WORLD);
	}
}

void recieve_blocks(blocks_t &blocks, int m, int n) {
	MPI_Status stat;
	long nnz;
	int procid = getprocid();
	MPI_Recv(&nnz,1,MPI_LONG,ROOT,9,MPI_COMM_WORLD,&stat);
	blocks.from_mpi(m, n, nnz);
	MPI_Recv(&blocks.allrates[0],sizeof(rate_t)*nnz,MPI_CHAR,ROOT,3,MPI_COMM_WORLD,&stat);
	MPI_Recv(&blocks.block_ptr[0],blocks.B+1,MPI_LONG,ROOT,9,MPI_COMM_WORLD,&stat);
	MPI_Recv(&blocks.nnz_row[0],blocks.rows,MPI_UNSIGNED,ROOT,9,MPI_COMM_WORLD,&stat);
	MPI_Recv(&blocks.nnz_col[0],blocks.cols,MPI_UNSIGNED,ROOT,9,MPI_COMM_WORLD,&stat);
	long shift = blocks.block_ptr[0]; 
	for(int i=0;i<=blocks.B;++i) blocks.block_ptr[i] -= shift;

}

/*
 * file_id (4 bytes int), number of rows (4 bytes int), number of columns (4 bytes int), number of nnzs (4 bytes int if in PETSc format, 8 bytes long long if file_id is 1015), number of nzs per row (4 bytes int * number of rows), column indices of nonzero values (4 bytes int * number of nnzs), values of nonzeros (8 bytes double * number of nnzs).
 * */

#define MAT_FILE_CLASSID 1211216 
#define LONG_FILE_CLASSID 1015 

class PETSc_reader{
	public:
		const char *filesrc;
		int rows, cols;
		long nnz;  // nnz of the entire data, instead of the nnz of the local data
		long headersize;
		vector<int> nnz_row;
		vector<long> row_ptr;
		PETSc_reader(const char*src): filesrc(src) {
			FILE *fp = fopen(filesrc=src,"r");
			headersize = 3*sizeof(int);
			int filetype;
			fread(&filetype, sizeof(int), 1, fp);
			fread(&rows, sizeof(int), 1, fp);
			fread(&cols, sizeof(int), 1, fp);
			if(filetype == MAT_FILE_CLASSID) {
				int tmp;
				headersize += sizeof(int)*fread(&tmp, sizeof(int), 1, fp);
				nnz = (long)tmp;
			} else if(filetype == LONG_FILE_CLASSID) {
				headersize += sizeof(int64_t)*fread(&nnz, sizeof(int64_t), 1, fp);
			} else {
				fprintf(stderr,"Wrong file type\n!");
			}
			nnz_row.resize(rows);
			row_ptr.resize(rows+1);
			headersize += sizeof(int)*fread(&nnz_row[0], sizeof(int), rows, fp);
			row_ptr[0] = 0;
			for(int i = 1; i <= rows; i++)
				row_ptr[i] = row_ptr[i-1] + nnz_row[i-1];
			fclose(fp);
		}

		// assume that procid starts from 1 to B
		void receive_blocks(int B, int innerB, int procid, mpi_blocks_t &blocks) {
			//blocks = mpi_blocks_t(B, innerB);
			blocks.set_parameters(B, innerB, rows, cols, nnz, block_row_ptr[procid]); // global nnz

			int start = block_row_ptr[procid];
			int end = block_row_ptr[procid+1];
			blocks.nnz = row_ptr[end] - row_ptr[start];
			//printf("procid %d: blocks.nnz %ld\n", procid, blocks.nnz);
			vector<double> val_t(blocks.nnz);
			vector<int> col_idx(blocks.nnz);
			FILE *fp = fopen(filesrc,"r");
			fseek(fp, headersize+row_ptr[start]*sizeof(int), SEEK_SET);
			fread(&col_idx[0], sizeof(int), blocks.nnz, fp);
			fseek(fp, headersize+nnz*sizeof(int)+row_ptr[start]*sizeof(double), SEEK_SET);
			fread(&val_t[0], sizeof(double), blocks.nnz, fp);
			fclose(fp);

			for(int row = 0; row < rows; row++)
				blocks.nnz_row[row] = row_ptr[row+1] - row_ptr[row];
			blocks.allrates.resize(blocks.nnz);
			for(int bid = 0; bid <= B*innerB*innerB; bid++)
				blocks.block_ptr[bid] = 0;
			size_t shift = row_ptr[start];
			for(int row = start; row < end; row++) {
				for(size_t idx = row_ptr[row]; idx != row_ptr[row+1]; idx++) {
					int bid = blocks.bid_of_rate(row, col_idx[idx-shift]);
					blocks.block_ptr[bid+1]++;
				}
			}
			for(int bid = 2; bid <= B*innerB*innerB; bid++)
				blocks.block_ptr[bid] += blocks.block_ptr[bid-1];
			for(int row = start; row < end; row++) {
				for(size_t idx = row_ptr[row]; idx != row_ptr[row+1]; idx++) {
					int bid = blocks.bid_of_rate(row, col_idx[idx-shift]);
					rate_t r(row, col_idx[idx-shift], val_t[idx-shift]);
					blocks.allrates[blocks.block_ptr[bid]++] = r;
					blocks.nnz_col[r.j]++;
				}
			}

			for(int bid = B*innerB*innerB; bid > 0; bid--)
				blocks.block_ptr[bid] = blocks.block_ptr[bid-1];
			blocks.block_ptr[0] = 0;

			// Get the nnz_col
			vector<unsigned> tmp(cols, 0);
			MPI_Allreduce(&blocks.nnz_col[0], &tmp[0], cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			blocks.nnz_col = tmp;

		}
		// Deprecated.
		void receive_blocks(int B, int procid, blocks_t &blocks) {
			blocks.from_mpi(rows, cols, 0);
			if(procid==ROOT) {
				blocks.nnz = nnz;
			} else {
				int start = block_row_ptr[procid]; //(procid-1)*blocks.Bm;
				int end = block_row_ptr[procid+1]; // std::min(start + blocks.Bm, rows);
				// nnz of the local data
				blocks.nnz = row_ptr[end] - row_ptr[start];
				vector<double> val_t(blocks.nnz);
				vector<int> col_idx(blocks.nnz);
				FILE *fp = fopen(filesrc,"r");
				fseek(fp, headersize+row_ptr[start]*sizeof(int), SEEK_SET);
				fread(&col_idx[0], sizeof(int), blocks.nnz, fp);
				fseek(fp, headersize+nnz*sizeof(int)+row_ptr[start]*sizeof(double), SEEK_SET);
				fread(&val_t[0], sizeof(double), blocks.nnz, fp);
				fclose(fp);

				for(int row = 0; row < rows; row++)
					blocks.nnz_row[row] = row_ptr[row+1] - row_ptr[row];

				blocks.allrates.resize(blocks.nnz);
				for(int bid = 0; bid <= B; bid++)
					blocks.block_ptr[bid] = 0;
				for(size_t idx = row_ptr[start], shift=row_ptr[start]; idx != row_ptr[end]; idx++)
					blocks.block_ptr[col_idx[idx-shift]/blocks.Bn+1]++;
				for(int bid = 2; bid <= B; bid++)
					blocks.block_ptr[bid] += blocks.block_ptr[bid-1];
				for(int row = start; row < end; row++) {
					for(size_t idx = row_ptr[row]; idx != row_ptr[row+1]; idx++) {
						rate_t r(row, col_idx[idx-row_ptr[start]], val_t[idx-row_ptr[start]]);
						int bid = r.j/blocks.Bn;
						blocks.allrates[blocks.block_ptr[bid]++] = r;
						blocks.nnz_col[r.j]++;
					}
				}
				for(int bid = B; bid > 0; bid--)
					blocks.block_ptr[bid] = blocks.block_ptr[bid-1];
				blocks.block_ptr[0] = 0;

			}
			// Get the nnz_col
			vector<unsigned> tmp(cols, 0);
			MPI_Allreduce(&blocks.nnz_col[0], &tmp[0], cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			blocks.nnz_col = tmp;
		}

};


// load all the text-fmt data from root
void load_2(const char *srcdir, blocks_t &blocks, blocks_t &testblocks, testset_t &T){
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp=fopen(filename,"r"); 
	long m, n, nnz;
	fscanf(fp,"%ld %ld", &m, &n, &nnz, buf); 

	fscanf(fp,"%ld %s", &nnz, buf); 
	sprintf(filename,"%s/%s", srcdir, buf);
	blocks.load(m, n, nnz, filename);
	sort(blocks.allrates.begin(), blocks.allrates.end(), RateComp(&blocks));

	if(fscanf(fp, "%ld %s", &nnz, buf)!= EOF){
		sprintf(filename,"%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
		testblocks.load(m, n, nnz, filename);
		sort(testblocks.allrates.begin(), testblocks.allrates.end(), RateComp(&testblocks));
	}
	fclose(fp);
	//double bias = blocks.get_global_mean(); blocks.remove_bias(bias); T.remove_bias(bias);
}

void initial_partial_H(mat_t &H, long k, int procid) {
  int begin = block_col_ptr[procid];
  int end = block_col_ptr[procid]+block_col_cnt[procid];
  
  rng_type rng(SEED_VALUE + procid * 131 + 139);
  std::uniform_real_distribution<> init_dist(0, 1.0/sqrt(k));
  
  for (int row_index=begin; row_index < end; row_index++) {
    for (int t=0; t < k; t++) {
      H[row_index][t] = init_dist(rng);
    }
  }
}

void initial_partial_W(mat_t &X, long n, long k, int procid, int nr_threads){
  int part_size = block_row_cnt[0]/nr_threads;
  std::uniform_real_distribution<> init_dist(0, 1.0/sqrt(k));

  X = mat_t(n, vec_t(0));

  for (int p=0; p < nr_threads; p++) {

    int start = block_row_ptr[procid] + p * part_size;
    int end;
    if (p < nr_threads - 1) {
      end = block_row_ptr[procid] + (p+1) * part_size;
    }
    else {
      end = block_row_ptr[procid] + block_row_cnt[procid];
    }
    
    printf("procid: %d, in initial_partial_W. p: %d, start: %d, end: %d\n", procid, p, start, end);

    rng_type rng(SEED_VALUE + procid * 131 + p + 1);

    vec_t tmp(k);
    srand(start);
    for(long i = start; i < end; i++) {
      for(long t = 0; t < k; t++)
	tmp[t] = init_dist(rng);
      X[i] = tmp;
    }

  }

}

// void initial_partial(mat_t &X, long n, long k, int procid){
//   int start = block_row_ptr[procid];
//   int end = start + block_row_cnt[procid];
//   X = mat_t(n, vec_t(0));
//   vec_t tmp(k);
//   srand(start);
//   for(long i = start; i < end; i++) {
//     for(long t = 0; t < k; t++)
//       tmp[t] = 0.1*drand48();
//     X[i] = tmp;
//   }
// }

void usage(){ 
	puts("export MV2_ENABLE_AFFINITY=0");
	puts("mpiexec -n 4 ./mpi-sgd-omp rank lambda maxiter lrate decay_rate strategy stratum_strategy nr_threads srcdir"); 
	puts("      strategy: 0 for fixed learning rate, 1 for bold driver"); 
	puts("      stratum_strategy: 0 for semi-WOR, 1 for WR, 2 for WOR"); 
}

int main(int argc, char* argv[]){
	if(argc != 10) { usage(); return -1;}
	int k = atoi(argv[1]); // rank
	double lambda = atof(argv[2]);
	int maxiter = atoi(argv[3]);
	double lrate = atof(argv[4]);
	double decay_rate = atof(argv[5]);
	int strategy = atoi(argv[6]);
	int stratum_strategy = atoi(argv[7]);
	int nr_threads = atoi(argv[8]);
	char* src = argv[9];
	long m, n, nnz;
	int Rsize[2];
	MPI_Init(&argc, &argv);
	int procid = getprocid();
	int numprocs = size(); // 0 for root, 1 ~ numprocs for slave
	int B = numprocs;
	int innerB = 2*nr_threads;
	mpi_blocks_t blocks, testblocks;
	testset_t T;
	mat_t W,H;

	char train_src[1024], test_src[1024];
	sprintf(train_src, "%s/train.dat", src);
	sprintf(test_src, "%s/test.dat", src);

	PETSc_reader training(train_src);
	PETSc_reader test(test_src);


	// Root load the data
	m = training.rows, n = training.cols;
	gen_block_ptr(m, n, numprocs, nr_threads);
	gen_block2(k,numprocs);	

	double time = omp_get_wtime();
	training.receive_blocks(B, innerB, procid, blocks);
	//printf("procid %d: loading training done\n", procid);
	test.receive_blocks(B, innerB, procid, testblocks);
	//printf("procid %d: loading test done\n", procid);

	printf("procid %d: loading done\n", procid);

	initial_partial_W(W, blocks.rows, k, procid, nr_threads);

	printf("procid %d: initial_partial_W done \n", procid);

	H = mat_t(blocks.cols, vec_t(k,0));
	initial_partial_H(H, k, procid);

	printf("procid %d: initial_partial_H done \n", procid);

	//printf("procid %d: initialization done.\n",procid);

	if(0) {
		// load test ratings into root, it is for debug
		char buf[1024], filename[1024];
		long mm, nn, nnznnz;
		sprintf(buf, "%s/meta", src);
		FILE *fp = fopen(buf,"r");
		fscanf(fp,"%ld %ld", &mm, &nn); 
		fscanf(fp,"%ld %s", &nnznnz, buf); 
		fscanf(fp,"%ld %s", &nnznnz, buf); 
		sprintf(filename,"%s/%s", src, buf);
		if(procid==ROOT)
			T.load(m, n, nnznnz, filename);
	}

	omp_set_num_threads(nr_threads);
	//printf("procid %d: created %d threads for innerB %d\n", procid, nr_threads, innerB);
	MPI_Barrier(MPI_COMM_WORLD);
	if(procid==ROOT)
		printf("Proc %d: data loading %.6g sec.\n", procid, omp_get_wtime()-time);
	dsgd_mpi(blocks, testblocks, W, H, k, lambda, lrate, decay_rate, strategy, stratum_strategy, T, maxiter);
	MPI_Finalize();

	return 0;
}

