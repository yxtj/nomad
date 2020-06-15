
#include "pmf-mpi.h"
#define kind dynamic,500
//global variables for MPI

#include <random>
typedef std::mt19937_64 rng_type;

#define SEED_VALUE 12345

using namespace std;
vector<int> block_row_ptr, block_col_ptr, block_row_cnt, block_col_cnt;

#define ROOT 0

void gen_block_ptr(int m, int n, int numprocs, int nr_thread) { //{{{
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
} //}}}

// ============  Input  ===============
// PETSc format
#define MAT_FILE_CLASSID 1211216 
#define LONG_FILE_CLASSID 1015 

class PETSc_reader{//{{{
	public:
		const char *filesrc;
		int rows, cols;
		long nnz;  // nnz of the entire data, instead of the nnz of the local data
		long headersize;
		vector<int> nnz_row;
		vector<long> row_ptr;
		PETSc_reader(const char *src): filesrc(src) {
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

		void clear_space(){
			nnz_row.resize(0);
			row_ptr.resize(0);
			vector<int>(nnz_row).swap(nnz_row);
			vector<long>(row_ptr).swap(row_ptr);
		}

		void receive_rows(int procid, smat_t &R, vector<int> *col_perm = NULL) {
			// row indices to load from the file
			int start = block_row_ptr[procid];
			int end = block_row_ptr[procid+1];
			R.rows = rows;
			R.cols = cols;

			// Read CSR from the binary PETSc format
			R.nnz = row_ptr[end] - row_ptr[start];
			R.val_t = MALLOC(double,R.nnz);
			R.col_idx = MALLOC(unsigned,R.nnz);
			R.row_ptr = MALLOC(long,R.rows+1);

			FILE *fp = fopen(filesrc,"r");
			fseek(fp, headersize+row_ptr[start]*sizeof(int), SEEK_SET);
			fread(&R.col_idx[0], sizeof(int), R.nnz, fp);
			fseek(fp, headersize+nnz*sizeof(int)+row_ptr[start]*sizeof(double), SEEK_SET);
			fread(&R.val_t[0], sizeof(double), R.nnz, fp);
			fclose(fp);
			memset(R.row_ptr, 0, sizeof(long)*(R.rows+1));
			if(col_perm != NULL) {
				for(size_t idx = 0; idx < R.nnz; idx++)
					R.col_idx[idx] = col_perm->at(R.col_idx[idx]);
			}
			for(int r = start; r <= end; r++)
				R.row_ptr[r] = row_ptr[r] - row_ptr[start];
			for(int r = end+1; r <= rows; r++)
				R.row_ptr[r] = R.row_ptr[end];

			// Convert CSR to CSC 
			R.val = MALLOC(double,R.nnz);
			R.row_idx = MALLOC(unsigned,R.nnz);
			R.col_ptr = MALLOC(long,R.cols+1);
			long *col_ptr = R.col_ptr; 
			memset(col_ptr, 0, sizeof(long)*(R.cols+1));
			for(long idx = 0; idx < R.nnz; idx++)
				col_ptr[R.col_idx[idx]+1]++;
			for(unsigned c = 1; c <= R.cols; c++)
				col_ptr[c] += col_ptr[c-1];
			for(unsigned r = start; r < end; r++){
				for(long idx = R.row_ptr[r]; idx != R.row_ptr[r+1]; idx++){
					long c = R.col_idx[idx];
					R.row_idx[col_ptr[c]] = r;
					R.val[col_ptr[c]++] = R.val_t[idx];
				}
			}
			for(long c = R.cols; c > 0; c--)
				col_ptr[c] = col_ptr[c-1];
			col_ptr[0] = 0;
			R.from_mpi();
		}
}; //}}}

void send_columns(smat_t &R, int c_start, int c_end, int dest_procid, MPI_Request *req){//{{{
	long nnz = R.col_ptr[c_end] - R.col_ptr[c_start];
	/*
	MPI_Send(&nnz, 1, MPI_LONG, dest_procid, dest_procid, MPI_COMM_WORLD);
	MPI_Send(&R.val[R.col_ptr[c_start]],nnz,MPI_DOUBLE,dest_procid,dest_procid,MPI_COMM_WORLD);
	MPI_Send(&R.row_idx[R.col_ptr[c_start]],nnz,MPI_UNSIGNED,dest_procid,dest_procid,MPI_COMM_WORLD);
	MPI_Send(&R.col_ptr[c_start],c_end-c_start+1,MPI_LONG,dest_procid,dest_procid,MPI_COMM_WORLD);
	*/
	MPI_Isend(&nnz, 1, MPI_LONG, dest_procid, dest_procid, MPI_COMM_WORLD, &req[0]);
	MPI_Isend(&R.val[R.col_ptr[c_start]],nnz,MPI_DOUBLE,dest_procid,dest_procid,MPI_COMM_WORLD, &req[1]);
	MPI_Isend(&R.row_idx[R.col_ptr[c_start]],nnz,MPI_UNSIGNED,dest_procid,dest_procid,MPI_COMM_WORLD, &req[2]);
	MPI_Isend(&R.col_ptr[c_start],c_end-c_start+1,MPI_LONG,dest_procid,dest_procid,MPI_COMM_WORLD, &req[3]);
}//}}}

void receive_columns(smat_t &R, int c_start, int c_end, int from_procid) { //{{{
	MPI_Status stat;
	MPI_Request req;
	int tag = get_procid();

	MPI_Recv(&R.nnz,1,MPI_LONG,from_procid,tag,MPI_COMM_WORLD,&stat);
	R.val = MALLOC(double,R.nnz); memset(R.val,0,sizeof(double)*R.nnz);
	R.row_idx = MALLOC(unsigned,R.nnz); memset(R.row_idx,0,sizeof(unsigned)*R.nnz);
	R.col_ptr = MALLOC(long,R.cols+1); memset(R.col_ptr,0,sizeof(long)*(R.cols+1));
	MPI_Recv(R.val,R.nnz,MPI_DOUBLE,from_procid,tag,MPI_COMM_WORLD,&stat);
	MPI_Recv(R.row_idx,R.nnz,MPI_UNSIGNED,from_procid,tag,MPI_COMM_WORLD,&stat);
	MPI_Recv(&R.col_ptr[c_start],c_end-c_start+1,MPI_LONG,from_procid,tag,MPI_COMM_WORLD,&stat);

	long shift = R.col_ptr[c_start];
	for(int c=c_start; c <= c_end; ++c) 
		R.col_ptr[c] -= shift;
	R.from_mpi();
}//}}}

// Distributed model 
class dist_model_t {//{{{
public:
  int rows, cols;
  int k;
  int procid;
  mat_t localW, localH;
  
  dist_model_t(int rows_, int cols_, int k_, int nr_thread): rows(rows_), cols(cols_), k(k_){
    procid = get_procid();
    initial_row(localW, k, block_row_cnt[procid], nr_thread);
    initial_col(localH, k, block_col_cnt[procid]);
  }

  void get_globalWt(int t, double *buf){
    MPI_Allgatherv(&localW[t][0], block_row_cnt[procid], MPI_DOUBLE, buf, 
		   &block_row_cnt[0], &block_row_ptr[0], MPI_DOUBLE, MPI_COMM_WORLD);
  }
  void get_globalHt(int t, double *buf){
    MPI_Allgatherv(&localH[t][0], block_col_cnt[procid], MPI_DOUBLE, buf, 
		   &block_col_cnt[0], &block_col_ptr[0], MPI_DOUBLE, MPI_COMM_WORLD);
  }
  double* get_localWt(int t) {return &localW[t][0];}
  double* get_localHt(int t) {return &localH[t][0];}
  void update_localWt(int t, double *buf){
    if(buf!=&localW[t][0])
      memcpy(&localW[t][0], buf, sizeof(double)*block_row_cnt[procid]);
  }
  void update_localHt(int t, double *buf){
    if(buf!=&localH[t][0])
      memcpy(&localH[t][0], buf, sizeof(double)*block_col_cnt[procid]);
  }
  
private:
  void initial_row(mat_t &X, long k, long n, int nr_threads){
    int part_size = block_row_cnt[0]/nr_threads;
    std::uniform_real_distribution<> init_dist(0, 1.0/sqrt(k));
    
    X = mat_t(k, vec_t(n));

    for (int p=0; p < nr_threads; p++) {

      rng_type rng(SEED_VALUE + procid * 131 + p + 1);

      int start = block_row_ptr[procid] + p * part_size;
      int end;
      if (p < nr_threads - 1) {
	end = block_row_ptr[procid] + (p+1) * part_size;
      }
      else {
	end = block_row_ptr[procid] + block_row_cnt[procid];
      }
      
      for(long i = 0; i < n; ++i)
	for(long j = 0; j < k; ++j)
	  X[j][i] = init_dist(rng);
    }
  }

  void initial_col(mat_t &X, long k, long n){
    double scale = 1./sqrt(k);
    X = mat_t(k, vec_t(n));
    srand48((long)procid);
    for(long i = 0; i < n; ++i)
      for(long j = 0; j < k; ++j)
	X[j][i] = scale*(2*drand48()-1.0); 
  }

};//}}}

// Distributed version of smat_t
class dist_smat_t{ // {{{
	public:
		int procid;
		long rows, cols, nnz;
		// Rs = Omega_{S_r}, Rg = \bar{\Omega}_{G_r} in paper
		smat_t Rs; // Omega_{S_r}
		smat_t Rg; // Omega_{G_r}
		dist_smat_t():rows(0),cols(0),nnz(0){}

		void load(PETSc_reader &reader, vector<int> *col_perm=NULL){
			procid = get_procid();
			rows = reader.rows;
			cols = reader.cols;
			nnz = reader.nnz;
			reader.receive_rows(procid, Rs, col_perm);
			convert_Rs_to_Rg();
		}
	private:
		void convert_Rs_to_Rg();
}; //}}}

void dist_smat_t::convert_Rs_to_Rg() {//{{{
	long rows = Rs.rows, cols = Rs.cols;

	// Get the nnz of col
	vector<unsigned>nnz_col(cols), tmp(cols);
	for(int c = 0; c < cols; c++) 
		tmp[c] = Rs.nnz_of_col(c);
	MPI_Allreduce(&tmp[0], &nnz_col[0], cols, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

	// Every process scatters columns to all other processes
	int nr_proc = nr_processors();
	int procid = get_procid();
	vector<MPI_Request> reqs(nr_proc*4, MPI_REQUEST_NULL);
	vector<MPI_Status> stats(nr_proc*4);
	for(int pid = 0; pid < nr_proc; pid++){
		if(pid == procid) continue;
		int c_start = block_col_ptr[pid];
		int c_end = block_col_ptr[pid+1];
		send_columns(Rs, c_start, c_end, pid, &reqs[pid*4]);
	}


	// Allocate space for Rg
	// column indices to load from the file
	int start = block_col_ptr[procid];
	int end = block_col_ptr[procid+1];
	Rg.rows = rows;
	Rg.cols = cols;
	long nnz = 0;
	for(long c = start; c < end; c++) 
		nnz += nnz_col[c];
	// Print column distribution
	//printf("procid %3d: start %5d end %5d Rg.nnz %11ld\n", procid, start, end, nnz);
	Rg.nnz = nnz;
	Rg.val = MALLOC(double, Rg.nnz);
	Rg.row_idx = MALLOC(unsigned, Rg.nnz);
	Rg.col_ptr = MALLOC(long, cols+1);
	long *col_ptr = Rg.col_ptr;
	memset(col_ptr, 0, sizeof(long)*(cols+1));
	for(int c = start+1; c <= end; c++)
		col_ptr[c] = col_ptr[c-1] + nnz_col[c-1];
	for(int c = end+1; c <= cols; c++)
		col_ptr[c] = col_ptr[end];

	// Every process gathers columns from all other processes
	for(int pid = 0; pid < nr_proc; pid++) {
		smat_t tmp;
		tmp.rows = rows;
		tmp.cols = cols;
		if(pid != procid) receive_columns(tmp, start, end, pid);
		smat_t &tmpR = (pid == procid)? Rs:tmp;
		
		for(int c = start; c < end; c++) {
			memcpy(&Rg.row_idx[col_ptr[c]], &tmpR.row_idx[tmpR.col_ptr[c]], sizeof(unsigned)*tmpR.nnz_of_col(c));
			memcpy(&Rg.val[col_ptr[c]], &tmpR.val[tmpR.col_ptr[c]], sizeof(double)*tmpR.nnz_of_col(c));
			col_ptr[c] += tmpR.nnz_of_col(c);
		}
	}

	// Re-adjust the col_ptr
	for(int c = end; c > start; c--)
		col_ptr[c] = col_ptr[c-1];
	col_ptr[start] = 0;

	Rg.from_mpi();
	MPI_Waitall(nr_proc*4, &reqs[0], &stats[0]);
}//}}}

// ============  Computation ===============
inline double RankOneUpdate(const smat_t &R, const int j, const vec_t &u, const double lambda, const double vj, double *redvar, int do_nmf){
	double g=0, h=lambda;
	if(R.col_ptr[j+1]==R.col_ptr[j]) return 0;
	for(long idx=R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx) {
		int i = R.row_idx[idx];
		g += u[i]*R.val[idx]; 
		h += u[i]*u[i];
	}
	double newvj = g/h, tmp = 0, delta = 0, fundec = 0;
	if(do_nmf>0 & newvj < 0) {
		newvj = 0;
		delta = vj; // old - new
		fundec = -2*g*vj; + h*vj*vj;
	} else {
		delta = vj - newvj;
		fundec = h*delta*delta;
	}
	//double delta = vj - newvj;
	//double fundec = h*delta*delta;
	//double lossdec = fundec - lambda*delta*(vj+newvj);
	//double gnorm = (g-h*vj)*(g-h*vj); 
	*redvar += fundec;
	//*redvar += lossdec;
	return newvj;
}

inline double UpdateRating(smat_t &R, const vec_t &Wt, const vec_t &Ht, bool add) {
	double loss=0;
	if(add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] +=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	} else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	}
}

void ccdr1_mpi(dist_smat_t &training_set, dist_smat_t &test_set, parameter &param, dist_model_t &model){
	int k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	double lambda = param.lambda;
	double eps = param.eps;
	int procid = get_procid();

	smat_t &R = training_set.Rg;
	smat_t Rt; Rt = training_set.Rs.transpose();
	smat_t &testR = test_set.Rg;
	smat_t testRt; testRt = test_set.Rs.transpose();

	vec_t u(R.rows), v(R.cols);
	double *localHt, *localWt;
	double localreg = 0, localloss = 0, localobj = 0, test_localloss = 0;

	double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	double starttime = 0, tmpstart, totaltime = 0, computetime = 0;
	long num_updates = 0;
	double reg=0,loss;

	int col_begin=block_col_ptr[procid], col_end=block_col_ptr[procid+1];
	int row_begin=block_row_ptr[procid], row_end=block_row_ptr[procid+1];

	// Initialization Stage
	// initial value of the regularization term
	// H is a zero matrix now.
	for(int t = 0 ; t < k; t++){
		localHt = model.get_localHt(t);
		localWt = model.get_localWt(t);
		memset(localHt, 0, sizeof(double)*block_col_cnt[procid]);
		for(long i = 0; row_begin+i < row_end; i++) 
			localreg += localWt[i]*localWt[i]*Rt.nnz_of_col(row_begin+i);
	}

	for(int oiter = 1; oiter <= maxiter; ++oiter) {
		double gnorm = 0, initgnorm=0;
		double rankfundec = 0;
		double fundec_max = 0;
		int early_stop = 0;

		for(int tt=0; tt < k; tt++) {
			int t = tt;
			if(early_stop >= 5) break;
			start = omp_get_wtime();
			starttime = omp_get_wtime();

			model.get_globalWt(t, &u[0]);
			model.get_globalHt(t, &v[0]);
			double *local_u = model.get_localWt(t);
			double *local_v = model.get_localHt(t);

			tmpstart = omp_get_wtime();
			// Create Rhat = R - Wt Ht^T
			if (oiter > 1) {
				// Assume H = 0 at the first iteration
				UpdateRating(R, u, v, true);
				UpdateRating(Rt, v, u, true);
				if(param.do_predict) {
					UpdateRating(testR, u, v, true);
					UpdateRating(testRt, v, u, true);
				}
			} 
			// Update localreg
			for(int i = 0; i < block_row_cnt[procid]; i++)
				localreg -= local_u[i]*local_u[i]*Rt.nnz_of_col(row_begin+i);
			for(int i = 0; i < block_col_cnt[procid]; i++)
				localreg -= local_v[i]*local_v[i]*R.nnz_of_col(col_begin+i);
			computetime += omp_get_wtime() - tmpstart;

			Itime += omp_get_wtime() - start;

			gnorm = 0, initgnorm=0;
			double innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter; 	
			for(int iter = 1; iter <= maxit; ++iter){
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				double fundec_cur = 0;
				tmpstart = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:fundec_cur)
				for(long c = col_begin; c < col_end; ++c)
					local_v[c-col_begin] = RankOneUpdate(R, c, u, lambda*R.nnz_of_col(c), 
							local_v[c-col_begin], &fundec_cur, param.do_nmf);
				computetime += omp_get_wtime() - tmpstart;
				model.get_globalHt(t, &v[0]);
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;

				// Update W[t]
				start = omp_get_wtime();
				tmpstart = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:fundec_cur)
				for(long r = row_begin; r < row_end; ++r)
					local_u[r-row_begin] = RankOneUpdate(Rt, r, v, lambda*Rt.nnz_of_col(r),
							local_u[r-row_begin], &fundec_cur, param.do_nmf);
				computetime += omp_get_wtime() - tmpstart;
				model.get_globalWt(t, &u[0]);
				num_updates += Rt.cols;

				MPI_Allreduce(&fundec_cur, &innerfundec_cur, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

				tmpstart = omp_get_wtime();
				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break; 
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = max(fundec_max, innerfundec_cur);
				computetime += omp_get_wtime() - tmpstart;
				Wtime += omp_get_wtime() - start;
			}

			// Update R and Rt
			start = omp_get_wtime();
			tmpstart = omp_get_wtime();
			model.update_localWt(t, local_u);
			model.update_localHt(t, local_v);
			localloss = UpdateRating(R, u, v, false);
			localloss = UpdateRating(Rt, v, u, false);
			computetime += omp_get_wtime() - tmpstart;

			totaltime += omp_get_wtime() - starttime;


			if(param.do_predict) {
				test_localloss = UpdateRating(testR, u, v, false);
				test_localloss = UpdateRating(testRt, v, u, false);
			}

			// Update localreg
			for(int i = 0; i < block_row_cnt[procid]; i++)
				localreg += local_u[i]*local_u[i]*Rt.nnz_of_col(row_begin+i);
			for(int i = 0; i < block_col_cnt[procid]; i++)
				localreg += local_v[i]*local_v[i]*R.nnz_of_col(col_begin+i);
			Rtime += omp_get_wtime() - start;


			double obj = 0, loss = 0, reg = 0;
			localobj = localloss + lambda*localreg;
			MPI_Allreduce(&localobj, &obj, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&localloss, &loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&localreg, &reg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			obj = loss + reg*lambda;

			double avg_totaltime = 0, avg_computetime = 0, max_computetime = 0, min_computetime = 1e500;
			MPI_Reduce(&totaltime, &avg_totaltime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
			MPI_Reduce(&computetime, &avg_computetime, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
			MPI_Reduce(&computetime, &max_computetime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
			MPI_Reduce(&computetime, &min_computetime, 1, MPI_DOUBLE, MPI_MIN, ROOT, MPI_COMM_WORLD);
			avg_totaltime /= nr_processors(); avg_computetime /= nr_processors();

			if(param.verbose and procid == ROOT)
				printf("iter %d rank %d totaltime %.6g computetime %.6g idle %.2g%% obj %.6g loss %.6g reg %.6g diff %.6g updates %.8g ",
						oiter,t+1, avg_totaltime, avg_computetime, 100.0*(avg_totaltime-avg_computetime)/(avg_totaltime), obj, loss, reg, oldobj - obj, (double)num_updates);
			oldobj = obj;

			if(param.do_predict and test_set.nnz != 0){ 
				double testloss= 0;
				MPI_Allreduce(&test_localloss, &testloss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				if(procid == ROOT)
					printf("rmse %.10g", sqrt(testloss/test_set.nnz)); 
			}
			if(param.verbose and procid == ROOT) { puts(""); fflush(stdout);}
		}
	}
	omp_set_num_threads(num_threads_old);

}

void exit_with_help()
{
	if(get_procid() == ROOT)
		printf(
				"Usage: \n"
				"    export MV2_ENABLE_AFFINITY=0\n"
				"    mpiexec -n 4 mpi-ccdr1-omp [options] data_dir [model_filename]\n"
				"options:\n"
				"    -s type : set type of solver (default 0)\n"    
				"    	 0 -- CCDR1 with fundec stopping condition\n"    
				"    -k rank : set the rank (default 10)\n"    
				"    -n threads : set the number of threads (default 4)\n"    
				"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
				"    -t max_iter: set the number of iterations (default 5)\n"    
				"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
				"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
				"    -p do_predict: do prediction or not (default 1)\n" // different from omp version
				"    -q verbose: show information or not (default 0)\n"
				"    -S do_random_suffle : conduct a random shuffle for columns (default 0)\n"     
				"    -N do_nmf: do nmf (default 0)\n"
			  );
	exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	parameter param;   // default values have been set by the constructor 
	int i;

	param.do_predict = 1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 'r':
				param.rho = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				param.eta0 = atof(argv[i]);
				break;

			case 'B':
				param.num_blocks = atoi(argv[i]);
				break;

			case 'm':
				param.lrate_method = atoi(argv[i]);
				break;

			case 'u':
				param.betaup = atof(argv[i]);
				break;

			case 'd':
				param.betadown = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'S':
				param.do_suffle = atoi(argv[i]) == 1? true : false;
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0) 
		param.verbose = 1;

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') 
			*p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	return param;
}

int main(int argc, char* argv[]){
	char input_file_name[1024], model_file_name[1024], train_src[1024], test_src[1024];

	MPI_Init(&argc, &argv);
	parameter param = parse_command_line(argc, argv, input_file_name, model_file_name); 

	double starttime = omp_get_wtime();
	sprintf(train_src, "%s/train.dat", input_file_name);
	sprintf(test_src, "%s/test.dat", input_file_name);

	PETSc_reader training_reader(train_src);
	PETSc_reader test_reader(test_src);
	dist_smat_t training_set;
	dist_smat_t test_set;

	// load the data
	int procid = get_procid();
	int numprocs = nr_processors();
	long m = training_reader.rows, n = training_reader.cols;
	gen_block_ptr(m, n, numprocs, param.threads);

	// Random shuffle of columns
	
	vector<int> col_perm(0);
	if(param.do_suffle) {
		srand(0);
		col_perm.resize(n);
		for(int c = 0; c < n; c++) col_perm[c] = c;
		for(int c = 0; c < n; c++) {
			int c2 = rand()%(n - c) + c;
			std::swap(col_perm[c], col_perm[c2]);
		}
	}
	training_set.load(training_reader, col_perm.size()==0? NULL : &col_perm);
	test_set.load(test_reader, col_perm.size()==0? NULL : &col_perm);

	training_reader.clear_space();
	test_reader.clear_space();

	MPI_Barrier(MPI_COMM_WORLD);
	if(procid==ROOT) printf("data loading done in %.4g..\n", omp_get_wtime() - starttime);

	omp_set_num_threads(param.threads);
	double time = omp_get_wtime();

	dist_model_t model(m, n, param.k, param.threads);

	ccdr1_mpi(training_set, test_set, param, model);
	MPI_Finalize();

	return 0;
}

