//#include "util.h"
#include "util-mpi.h"
#define kind dynamic

bool dryrun = false;
bool nonblocking = true;

#include <random>
#define SEED_VALUE 12345
typedef std::mt19937_64 rng_type;

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
struct range_t {
	int begin, end;
	range_t(int b=0, int e=0): begin(b), end(e){}
	int size() {return end-begin;}
};

class mpi_blocks_t{
	public:
		mpi_blocks_t(){}
		// Create a B*B blocks, each block has innerB-by-2innerB subblocks
		void set_parameters(int _B, int _innerB, int _rows, int _cols, long _global_nnz, int _startrow) {
			B = _B; innerB = _innerB;
			rows = _rows; cols = _cols; 
			global_nnz = _global_nnz; startrow = _startrow;
			Bm = rows/B+((rows%B)?1:0); // block's row size 
			Bn = cols/B+((cols%B)?1:0); // block's col size
			innerBm = Bm/innerB+((Bm%innerB)?1:0); // innerblock's row size
			innerBn = Bn/(2*innerB)+((Bn%(2*innerB))?1:0); // innerblock's col size
			nnz_row.resize(rows);
			nnz_col.resize(cols);
			block_ptr.resize(2*B*innerB*innerB+1);
		}
		void compressed_space(){
			vector<long>(block_ptr).swap(block_ptr);
			vector<rate_t>(allrates).swap(allrates);
			vector<unsigned>(nnz_row).swap(nnz_row);
			vector<unsigned>(nnz_col).swap(nnz_col);
			for(int bid = 1; bid <= 2*B*innerB*innerB; ++bid)
				block_ptr[bid] += block_ptr[bid-1];
		}
		inline int global_row(int localr) const {return startrow+localr;}
		inline int local_row(int r) const {return r-startrow;}
		inline int bid_of_rate(int r, int c) const {
			int out_bid = c/Bn; // outer block id
			int bi = (r-startrow)/innerBm; // (i, j) index for inner blocks
			int bj = (c-out_bid*Bn)/innerBn; 
			return out_bid*innerB*2*innerB + bi*2*innerB + bj;
		}

		range_t get_col_rng(int block_col_id, int phase=-1) const {
			int begin = block_col_ptr[block_col_id];
			int middle = begin + innerB*innerBn; 
			int end = begin + block_col_cnt[block_col_id];
			if (phase == 0) {
				return range_t(begin, middle);
			} else if (phase == 1) {
				return range_t(middle, end);
			} else {
				return range_t(begin, end);
			}
		}

		inline int size() const {return 2*B*innerB*innerB;}
		inline rateset_t operator[] (int bid) const {
			return rateset_t(&allrates[block_ptr[bid]], block_ptr[bid+1]-block_ptr[bid]);
		}
		inline rateset_t  getrateset(int out_bid, int bi, int bj) const {
			return (*this)[out_bid*2*innerB*innerB + bi*2*innerB + bj];
		}
		inline rateset_t getrateset(int bi, int bj) const {
			return (*this)[bi * B + bj]; 
		}
		inline rateset_t  getrateset(int out_bid, int phase, int bi, int bj) const {
			return (*this)[out_bid*2*innerB*innerB + bi*2*innerB + phase*innerB + bj];
		}
		inline int get_bid(int block_col_id, int phase, int bi, int bj) const {
			return block_col_id*2*innerB*innerB + bi*2*innerB + phase*innerB + bj;
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
void Two2One(mat_t &W, vec_t &W1, int k, range_t rng) { Two2One(W, W1, k, rng.begin, rng.end); }
void One2Two(vec_t &W1, mat_t &W, int k, range_t rng) { One2Two(W1, W, k, rng.begin, rng.end); }

struct sender_t {
	vec_t buf;
	int k;
	MPI_Request req;
	MPI_Status stat;
	double computetime;
	bool busy;
	sender_t(int k_, size_t buf_size): k(k_), busy(false), computetime(0){ buf.resize(buf_size); }
	void send_to(int dest, mat_t &H, range_t rng) {
		finish(); // make sure the sender is free now

		double tmpstart = omp_get_wtime();
		Two2One(H, buf, k, rng);
		computetime += omp_get_wtime() - tmpstart;

		MPI_Isend(&buf[0], k*rng.size(), MPI_DOUBLE, dest, dest, MPI_COMM_WORLD, &req);
		busy = true;
	}
	void finish(){ // wait for revious send to finish
		if(busy) {
			MPI_Wait(&req, &stat);
			busy = false;
		}
	}
	void reset_timer(){computetime = 0;}
};

struct receiver_t {
	vec_t buf;
	int k;
	MPI_Request req;
	MPI_Status stat;
	bool busy;
	double computetime;

	receiver_t(int k_, size_t buf_size): k(k_), busy(false){ buf.resize(buf_size); }
	bool recv_from(int src, range_t rng){
		if(busy) return false;
		MPI_Irecv(&buf[0], k*rng.size(), MPI_DOUBLE, src, getprocid(), MPI_COMM_WORLD, &req);
		busy = true;
		return true;
	}
	void finish_and_copy(mat_t &H, range_t rng){
		if(busy) {
			MPI_Wait(&req, &stat);
			busy = false;
			double tmpstart = omp_get_wtime();
			One2Two(buf, H, k, rng);
			computetime += omp_get_wtime() - tmpstart;
		}  
	}
	void reset_timer(){computetime = 0;}
};


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
	if(computetime!=NULL) *computetime += omp_get_wtime() - tmpstart;

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
	for(int id = 2*blocks.B*blocks.innerB*blocks.innerB-1; id >=0; --id){
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

void dsgdpp_mpi(const mpi_blocks_t &blocks, const mpi_blocks_t &testblocks, mat_t &W, mat_t &H, 
		int k, double lambda, double lrate, double decay_rate, int strategy, int stratum_strategy, testset_t &T, int maxiter=10){
	long B = blocks.B, m = blocks.rows, n = blocks.cols;
	long innerB = blocks.innerB;
	long nnz = blocks.nnz;
	int procid = getprocid();
	double obj, oldobj;
	double time = 0, starttime, tmpstart, transtime=0;
	double totaltime = 0, computetime = 0, objtime = 0;
	double cur_lrate = lrate;
	int sync_signal;
	vec_t localW1, localH1; // for slave node
	vec_t W1, H1;
	vec_t send_buf(blocks.Bn*k);
	vec_t recv_buf(blocks.Bn*k);
	H1.resize(n*k);
	localH1.resize(blocks.Bn*k);

	vector<sender_t> sender(2, sender_t(k, blocks.Bn*k));
	vector<receiver_t> receiver(2, receiver_t(k, blocks.Bn*k));

	MPI_Status stat;

	vector<vector<int> > perm_buffer(2, vector<int>(B));
	vector<int> perm(B); // column permutation 
	vector<int> offset(B); // row permutation 
	vector<int> permuted_block_col_ptr_k(B+1);
	vector<int> permuted_block_col_cnt_k(B+1);
	vector<int> inner_cur_perm(innerB);
	vector<int> inner_perm(innerB);
	vector<int> inner_offset(innerB);

	range_t recv_rng, col_rng;
	size_t updatecnt = 0;


	// Time to calculate the initial obj is not counted 
	if (strategy == BOLDDRIVER) {
		// obj is required for bold-driver
		starttime = omp_get_wtime();
		oldobj = calobj_mpi(blocks, W, H, lambda, &transtime);
		objtime += omp_get_wtime() - starttime;
		if(procid==ROOT) printf("initial obj %.6g\n", oldobj);
	}

	// Because of MPI_AllGather everybody has the freshest copy of the entire H
	for(int iter = 1; iter <= maxiter; ++iter){
		MPI_Barrier(MPI_COMM_WORLD);
		starttime = omp_get_wtime();
		sender[0].reset_timer();
		sender[1].reset_timer();
		receiver[0].reset_timer();
		receiver[1].reset_timer();

		// Initialize permuation arrays {{{
		tmpstart = omp_get_wtime();
		int *cur_perm = &perm_buffer[0][0], *next_perm = &perm_buffer[1][0];
		for(int i = 0; i < B; i++) perm[i] = offset[i] = cur_perm[i] = next_perm[i] = i;
		for(int i = 0; i < innerB; i++) inner_perm[i] = inner_offset[i] = inner_cur_perm[i] = i;
		unsigned seed = 34586U + (unsigned)(iter); 
		std::srand(seed);
		if(stratum_strategy == WOR_STRATUM) {
			std::random_shuffle(perm.begin(), perm.end());
			std::random_shuffle(offset.begin(), offset.end());
		}

		if (strategy == FIXED)
			cur_lrate = 1.5*lrate/(1.0+decay_rate*pow(iter, 1.5));

		// initialization for cur_perm
		for(int ss = 0; ss < B; ss++)
			cur_perm[perm[ss]] = (ss+offset[0])%B;

		computetime += omp_get_wtime() - tmpstart; // }}}

		for(int s = 0; s < B; ++s) {
			// sample a stratum for next subepoch {{{
			tmpstart = omp_get_wtime(); 
			int send_procid, recv_procid;
			if(stratum_strategy == SEMIWOR_STRATUM or stratum_strategy == WOR_STRATUM) {
				for(int ss = 0; ss < B; ss++)
					next_perm[perm[ss]] = (ss+offset[(s+1)%B])%B;
			} else if(stratum_strategy == WR_STRATUM) {
				unsigned seed = (unsigned)(iter*B+s); 
				for(int ss = 0; ss < B; ss++) {
					int j = ss + rand_r(&seed) % (B - ss);
					std::swap(next_perm[ss], next_perm[j]);
				}
			}
			// Locate the send and recv proc ID
			for(int ss = 0; ss < B; ss++) {
				if(next_perm[ss] == cur_perm[procid]) send_procid = ss;
				if(next_perm[procid] == cur_perm[ss]) recv_procid = ss;
			}
			int block_col_id = cur_perm[procid]; 
			computetime += omp_get_wtime() - tmpstart; // }}}

			int phase = 0;
			col_rng = blocks.get_col_rng(block_col_id, phase);
			
			if(s>0) { // make sure the communication is done
				if(nonblocking)	receiver[phase].finish_and_copy(H,col_rng);
			}

			// Computation for phase 0 {{{
			tmpstart = omp_get_wtime();
			// random stratum for inner blocks
			if(stratum_strategy == WOR_STRATUM) {
				std::random_shuffle(inner_perm.begin(), inner_perm.end());
				std::random_shuffle(inner_offset.begin(), inner_offset.end());
			}
			for(int i = 0; i < innerB; i++) {
				for(int ii = 0; ii < innerB; ii++)
					inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(kind) reduction (+:updatecnt)
				for(int bi = 0; bi< innerB; bi++) {
					unsigned seed = bi * iter;
					//int bj = (bi+i)%innerB;
					int bj = inner_cur_perm[bi];
					//int bid = block_col_id*innerB*innerB+bi*innerB+bj;
					//sgd(blocks[bid], W, H, k, lambda, cur_lrate, &seed, 1);
					rateset_t sub_block = blocks.getrateset(block_col_id, phase, bi, bj);
					sgd(sub_block, W, H, k, lambda, cur_lrate, &seed, 1);
					updatecnt += sub_block.size();
				}
			}
			// Make nonblocking send/recv requests
			if(not dryrun)
				if(s < B-1)  {
					recv_rng = blocks.get_col_rng(cur_perm[recv_procid], phase);
					sender[phase].send_to(send_procid, H, col_rng); 
					receiver[phase].recv_from(recv_procid, recv_rng);
				}
			computetime += omp_get_wtime() - tmpstart;

			if(not dryrun)
				if(not nonblocking and s < B-1) receiver[phase].finish_and_copy(H, recv_rng);
			// }}}

			phase = 1;
			col_rng = blocks.get_col_rng(block_col_id, phase);
			if(s>0) { // Make sure the communication is done
				if(nonblocking) receiver[phase].finish_and_copy(H,col_rng);
			}

			// Computation for phase 1 {{{
			tmpstart = omp_get_wtime();
			// random stratum for inner blocks
			/*
			if(stratum_strategy == WOR_STRATUM) {
				std::random_shuffle(inner_perm.begin(), inner_perm.end());
				std::random_shuffle(inner_offset.begin(), inner_offset.end());
			}
			*/
			for(int i = 0; i < innerB; i++) {
				for(int ii = 0; ii < innerB; ii++)
					inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(kind) reduction (+:updatecnt)
				for(int bi = 0; bi< innerB; bi++) {
					unsigned seed = bi * iter;
					//int bj = (bi+i)%innerB;
					int bj = inner_cur_perm[bi];
					//int bid = block_col_id*innerB*innerB+bi*innerB+bj;
					//sgd(blocks[bid], W, H, k, lambda, cur_lrate, begin, end, &seed, 1);
					rateset_t sub_block = blocks.getrateset(block_col_id, phase, bi, bj);
					sgd(sub_block, W, H, k, lambda, cur_lrate, &seed, 1);
					updatecnt += sub_block.size();
				}
			}
			if(not dryrun)
				if(s < B-1)  {
					recv_rng = blocks.get_col_rng(cur_perm[recv_procid], phase);
					sender[phase].send_to(send_procid, H, col_rng); 
					receiver[phase].recv_from(recv_procid, blocks.get_col_rng(cur_perm[recv_procid], phase));
					if(not nonblocking) receiver[phase].finish_and_copy(H, recv_rng);
				}
			computetime += omp_get_wtime() - tmpstart;
			if(not dryrun)
				if(not nonblocking and s < B-1) receiver[phase].finish_and_copy(H, recv_rng);
			// }}}

			// cyclic H transfer
			if(s < B-1) std::swap(cur_perm, next_perm);
		}
		
		computetime += sender[0].computetime;
		computetime += receiver[0].computetime;
		computetime += sender[1].computetime;
		computetime += receiver[1].computetime;

		if(strategy != BOLDDRIVER) {
			MPI_Barrier(MPI_COMM_WORLD);
			totaltime += omp_get_wtime() - starttime;
		}

		// Allgather H
		tmpstart = omp_get_wtime();
		for(int ss = 0; ss < B; ss++) {
			permuted_block_col_ptr_k[ss] = block_col_ptr_k[cur_perm[ss]];
			permuted_block_col_cnt_k[ss] = block_col_cnt_k[cur_perm[ss]];
		}
		Two2One(H,localH1,k,blocks.get_col_rng(cur_perm[procid]));
		MPI_Allgatherv(&localH1[0],permuted_block_col_cnt_k[procid],MPI_DOUBLE, &H1[0], &permuted_block_col_cnt_k[0], &permuted_block_col_ptr_k[0],MPI_DOUBLE, MPI_COMM_WORLD);
		transtime += omp_get_wtime() - tmpstart;
		One2Two(H1,H,k,0,n);

		// BOLDDRIVER should include the obj calculation time
		if(strategy == BOLDDRIVER) {

			// Calculate obj to update learning rate for bold driver 
			double tmp_start = omp_get_wtime();
			obj = calobj_mpi(blocks, W, H, lambda, &transtime, &computetime);
			if(obj > oldobj) cur_lrate *= decay_rate; else cur_lrate *= 1.05;
			oldobj = obj;
			objtime += omp_get_wtime() - tmp_start;

			MPI_Barrier(MPI_COMM_WORLD);
			totaltime += omp_get_wtime() - starttime;
		}


		// Timing calculation
		double avg_totaltime = 0, avg_computetime = 0, max_computetime = 0, min_computetime = 1e200;
		MPI_Reduce(&totaltime, &avg_totaltime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &avg_computetime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &max_computetime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(&computetime, &min_computetime, 1, MPI_DOUBLE, MPI_MIN, ROOT, MPI_COMM_WORLD);
		avg_totaltime /= blocks.B; avg_computetime /= blocks.B;

		size_t total_cnt=0;
		MPI_Reduce(&updatecnt, &total_cnt, 1, MPI_LONG, MPI_SUM, ROOT, MPI_COMM_WORLD);

		if(procid==ROOT){
			printf("cnt %ld ", total_cnt);
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

void gen_block_ptr(int m, int n, int numprocs, int nr_thread=1) {

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
			for(int bid = 0; bid <= blocks.size(); bid++)
				blocks.block_ptr[bid] = 0;
			size_t shift = row_ptr[start];
			for(int row = start; row < end; row++) {
				for(size_t idx = row_ptr[row]; idx != row_ptr[row+1]; idx++) {
					int bid = blocks.bid_of_rate(row, col_idx[idx-shift]);
					blocks.block_ptr[bid+1]++;
				}
			}
			for(int bid = 2; bid <= blocks.size(); bid++)
				blocks.block_ptr[bid] += blocks.block_ptr[bid-1];
			for(int row = start; row < end; row++) {
				for(size_t idx = row_ptr[row]; idx != row_ptr[row+1]; idx++) {
					int bid = blocks.bid_of_rate(row, col_idx[idx-shift]);
					rate_t r(row, col_idx[idx-shift], val_t[idx-shift]);
					blocks.allrates[blocks.block_ptr[bid]++] = r;
					blocks.nnz_col[r.j]++;
				}
			}

			for(int bid = blocks.size(); bid > 0; bid--)
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

void initial_partial(mat_t &X, long n, long k, int procid){
	int start = block_row_ptr[procid];
	int end = start + block_row_cnt[procid];
	X = mat_t(n, vec_t(0));
	vec_t tmp(k);
	srand48((long)procid);
	double scale = 1./sqrt(k);
	for(long i = start; i < end; i++) {
		for(long t = 0; t < k; t++)
			tmp[t] = scale*(2*drand48()-1.0);
		X[i] = tmp;
	}
}
// Same initialization as NOMAD, requires C++11
void initial_entire_H(mat_t &H, long k, int procid) {
	for(int procid = 0; procid < size(); procid++) {
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
}

// Same initialization as NOMAD, requires C++11
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

	//	printf("procid: %d, in initial_partial_W. p: %d, start: %d, end: %d\n", procid, p, start, end);

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

void usage(){ 
	puts("export MV2_ENABLE_AFFINITY=0");
	puts("mpiexec -n 4 ./mpi-dsgdpp rank lambda maxiter lrate decay_rate strategy stratum_strategy nr_threads srcdir"); 
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
	gen_block_ptr(m, n, numprocs, nr_threads); // Same as NOMAD
	//gen_block_ptr(m, n, numprocs); // Old initialization
	gen_block2(k,numprocs);	

	double time = omp_get_wtime();
	training.receive_blocks(B, innerB, procid, blocks);
	test.receive_blocks(B, innerB, procid, testblocks);

	// Same as NOMAD
	initial_partial_W(W, blocks.rows, k, procid, nr_threads);
	H = mat_t(blocks.cols, vec_t(k,0));
	initial_entire_H(H, k, procid);

	// Old Initialization
	//initial_partial(W, blocks.rows, k, procid);
	//H = mat_t(blocks.cols, vec_t(k,0));
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
	dsgdpp_mpi(blocks, testblocks, W, H, k, lambda, lrate, decay_rate, strategy, stratum_strategy, T, maxiter);
	MPI_Finalize();

	return 0;
}

