#include "util-mpi.h"
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

// load utility for block matrix format
void load(const char *srcdir, blocks_t &blocks, testset_t &T){
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp=fopen(filename,"r"); 
	long m, n, nnz;
	fscanf(fp,"%ld %ld", &m, &n); 

	fscanf(fp,"%ld %s", &nnz, buf); 
	sprintf(filename,"%s/%s", srcdir, buf);
	blocks.load(m, n, nnz, filename);
	sort(blocks.allrates.begin(), blocks.allrates.end(), RateComp(&blocks));

	if(fscanf(fp, "%ld %s", &nnz, buf)!= EOF){
		sprintf(filename,"%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
	}
	fclose(fp);
	//double bias = blocks.get_global_mean(); blocks.remove_bias(bias); T.remove_bias(bias);
}


// load utility for CCS RCS
void load(const char* srcdir, smat_t &R, testset_t &T){
	// add testing later
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp = fopen(filename,"r");
	long m, n, nnz;
	fscanf(fp, "%ld %ld", &m, &n);

	fscanf(fp, "%ld %s", &nnz, buf);
	sprintf(filename,"%s/%s", srcdir, buf);
	R.load(m, n, nnz, filename);

	if(fscanf(fp, "%ld %s", &nnz, buf)!= EOF){
		sprintf(filename,"%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
	}
	fclose(fp);
	//double bias = R.get_global_mean(); R.remove_bias(bias); T.remove_bias(bias);
	return ;
}

void initial(mat_t &X, long n, long k){
	X = mat_t(n, vec_t(k));
	srand(0);
	for(long i = 0; i < n; ++i){
		for(long j = 0; j < k; ++j)
			X[i][j] = 1*drand48(); //-1;
			//X[i][j] = 0; //-1;
	}
}

void initial_col_ans(mat_t &X, long k, long n){
	X = mat_t(k, vec_t(n));
	srand(5);
	for(long i = 0; i < n; ++i)
		for(long j = 0; j < k; ++j)
			X[j][i] = 1*drand48(); //-1;
}


void initial_col(mat_t &X, long k, long n){
	X = mat_t(k, vec_t(n));
	srand(0);
	for(long i = 0; i < n; ++i)
		for(long j = 0; j < k; ++j)
			X[j][i] = 1*drand48(); //-1;
}

double dot(const vec_t &a, const vec_t &b){
	double ret = 0;
	for(int i = a.size()-1; i >=0; --i)
		ret+=a[i]*b[i];
	return ret;
}
double dot(const mat_t &W, const int i, const mat_t &H, const int j){
	int k = W.size();
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*H[t][j];
	return ret;
}
double dot(const mat_t &W, const int i, const vec_t &H_j){
	int k = H_j.size();
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*H_j[t];
	return ret;
}
double norm(const vec_t &a){
	double ret = 0;
	for(int i = a.size()-1; i >=0; --i)
		ret+=a[i]*a[i];
	return ret;
}
double norm(const mat_t &M) {
	double reg = 0;
	for(int i = M.size()-1; i>=0; --i) reg += norm(M[i]);
	return reg;
}
double calloss(const smat_t &R, const mat_t &W, const mat_t &H){
	double loss = 0;
	int k = H.size();
	for(long c = 0; c < R.cols; ++c){
		for(long idx = R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
			double diff = - R.val[idx];
			diff += dot(W[R.row_idx[idx]], H[c]);
			loss += diff*diff;
		}
	}
	return loss;
}
double calobj(const smat_t &R, const mat_t &W, const mat_t &H, const double lambda, bool iscol){
	double loss = 0;
	int k = iscol?H.size():0;
	vec_t Hc(k);
	for(long c = 0; c < R.cols; ++c){
		if(iscol) 
			for(int t=0;t<k;++t) Hc[t] = H[t][c];
		for(long idx = R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
			double diff = - R.val[idx];
			if(iscol)
				diff += dot(W, R.row_idx[idx], Hc);
			else 
				diff += dot(W[R.row_idx[idx]], H[c]);
			loss += diff*diff;
		}
	}
	double reg = 0;
	if(iscol) {
		for(int t=0;t<k;++t) {
			for(long r=0;r<R.rows;++r) reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
			for(long c=0;c<R.cols;++c) reg += H[t][c]*H[t][c]*R.nnz_of_col(c);
		}
	} else {
		for(long r=0;r<R.rows;++r) reg += R.nnz_of_row(r)*norm(W[r]);
		for(long c=0;c<R.cols;++c) reg += R.nnz_of_col(c)*norm(H[c]);
	}
	reg *= lambda;
	return loss + reg;
}

double calobj(const blocks_t &blocks, const mat_t &W, const mat_t &H, const double lambda, bool iscol){
	double loss = 0;
	for(int id = blocks.size()-1; id >=0; --id){
		const rateset_t &rateset = blocks[id];
		for(int idx = rateset.size(); idx >=0; --idx){
			double diff = -rateset[idx].v;
			if(iscol)
				diff += dot(W, rateset[idx].i, H, rateset[idx].j);
			else 
				diff += dot(W[rateset[idx].i], H[rateset[idx].j]);
			loss += diff*diff;
		}
	}
	double reg = 0;
	if(iscol) {
		int k = H.size();
		for(int t=0;t<k;++t) {
			for(long r=0;r<blocks.rows;++r) reg += W[t][r]*W[t][r]*blocks.nnz_of_row(r);
			for(long c=0;c<blocks.cols;++c) reg += H[t][c]*H[t][c]*blocks.nnz_of_col(c);
		}
	} else {
		for(long r=0;r<blocks.rows;++r) reg += blocks.nnz_of_row(r)*norm(W[r]);
		for(long c=0;c<blocks.cols;++c) reg += blocks.nnz_of_col(c)*norm(H[c]);
	}
	return loss + lambda*reg;
}
double calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for(size_t idx = 0; idx < nnz; ++idx){
		err = -testset[idx].v;
		if(iscol)
			err += dot(W, testset[idx].i, H, testset[idx].j);
		else 
			err += dot(W[testset[idx].i], H[testset[idx].j]);
		rmse += err*err;
	}
	return sqrt(rmse/nnz);
}

double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse/nnz);
}

double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j] - oldWt[testset[idx].i]*oldHt[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse/nnz);
}

/*
void updateOne(vec_t &Hj, smat_t &R, long j, mat_t &W, double lambda, int maxiter=1){
	int k = Hj.size();
	long nnz = R.col_ptr[j+1] - R.col_ptr[j];
	if(!nnz) return ;
	double *err = MALLOC(double,nnz);
	int *t_idx = MALLOC(int, k);
	for(int t = 0; t < k; ++t) t_idx[t] = t;
	for(long idx = R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx){
		long i = R.row_idx[idx];
		err[idx - R.col_ptr[j]] = dot(W[i], Hj) - R.val[idx];
	}
	unsigned int seed = 0;
	while(maxiter--){
		for(int tt = 0; tt < k; tt++) {
			int ss = tt+rand_r(&seed)%(k-tt);
			swap(t_idx[tt],t_idx[ss]);
			int t = t_idx[tt];
			double g = lambda*Hj[t], h = lambda;
			for(long idx = R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx){
				double Wit = W[R.row_idx[idx]][t];
				g += Wit * err[idx - R.col_ptr[j]];
				h += Wit * Wit;
			}
			double Hjt_diff = -g/h;
			Hj[t] += Hjt_diff;
			for(long idx = R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx)
				err[idx-R.col_ptr[j]] +=  Hjt_diff * W[R.row_idx[idx]][t];
		}
	}

	free(err);
	free(t_idx);
	fflush(stdout);
}
// Parallel Coordinate Descent for Matrix Factorization
void pcd(smat_t &R, mat_t &W, mat_t &H, double lambda, testset_t &T, int maxiter=10){
	for(int iter = 1; iter <= maxiter; ++iter){
		// Update H
		for(long c = 0; c < R.cols; ++c)
			updateOne(H[c], R, c, W, lambda);
		smat_t Rt;
		Rt = R.transpose();
		// Update M
		for(long c = 0; c < Rt.cols; ++c)
			updateOne(W[c], Rt, c, H, lambda);
		printf("iter %d obj %.6g ", iter, calobj(R, W, H, lambda));
		if(T.nnz!=0){
			printf("rmse %.6g", calrmse(T, W, H));
		}
		puts("");
	}

}

int main(int argc, char* argv[]){
	char* src = argv[2];
	int k = atoi(argv[1]); // rank
	double lambda = 0.1;
	smat_t R;
	mat_t W,H;
	testset_t T;
	load(src,R,T);
	initial(W, R.rows, k);
	initial(H, R.cols, k);
	printf("Obj = %.3g\n", calobj(R,W,H,lambda));
	pcd(R,W,H,lambda,T,3);

	return 0;
}

*/
