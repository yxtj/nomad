#include "nomad_body.h"

#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <glog/logging.h>

#if defined(WIN32) || defined(_WIN32)
#undef min
#undef max
#endif // WIN32

#define MAT_FILE_CLASSID 1211216    /* used to indicate matrices in binary files */
#define LONG_FILE_CLASSID 1015    /* used to indicate matrices in binary files with large number of nonzeroes */

using namespace std;

static bool read_data(const string filename, int part_index, int num_parts,
	int& min_row_index,
	int& local_num_rows,
	long long& local_num_nonzero,
	vector<int, sallocator<int> >& col_offset,
	vector<int, sallocator<int> >& row_idx,
	vector<double, sallocator<double> >& row_val,
	int& nrows, int& ncols, long long& total_nnz
)
{

	ifstream data_file(filename, ios::in | ios::binary);

	int file_id;

	// read the ID of the file to figure out whether it is normal file format
	// or long file format
	if(!data_file.read(reinterpret_cast<char*>(&file_id), sizeof(int))){
		LOG(ERROR) << "Error in reading ID from file" << endl;
		return false;
	}

	// in this case, the file is in regular PETSc foramt
	if(file_id == MAT_FILE_CLASSID){
		int header[3];

		if(!data_file.read(reinterpret_cast<char*>(header), 3 * sizeof(int))){
			LOG(ERROR) << "Error in reading header from file" << endl;
			return false;
		}

		nrows = header[0];
		ncols = header[1];
		total_nnz = header[2];
	}
	// in this case, it is in PETSc format as well, but the nnz is in long long.
	else if(file_id == LONG_FILE_CLASSID){
		int header[2];

		if(!data_file.read(reinterpret_cast<char*>(header), 2 * sizeof(int))){
			LOG(ERROR) << "Error in reading header from file" << endl;
			return false;
		}

		nrows = header[0];
		ncols = header[1];

		if(!data_file.read(reinterpret_cast<char*>(&total_nnz), sizeof(long long))){
			LOG(ERROR) << "Error in reading nnz from file" << endl;
			return false;
		}

	} else{
		LOG(ERROR) << file_id << " does not identify a valid binary matrix file!" << endl;
		exit(1);
	}

	//LOG(INFO) << "nrows: " << nrows << ", ncols: " << ncols << ", total_nnz: " << total_nnz << endl;

	// calculate how many number of rows is to be stored locally
	const int num_rows_per_part = nrows / num_parts + ((nrows % num_parts > 0) ? 1 : 0);
	const int min_row = num_rows_per_part * part_index;
	min_row_index = min_row;
	const int max_row = std::min(num_rows_per_part * (part_index + 1), nrows);

	// return the number of rows stored in the machine, by reference
	local_num_rows = max_row - min_row;

	int* total_nnz_rows = sallocator<int>().allocate(nrows);
	if(!data_file.read(reinterpret_cast<char*>(total_nnz_rows), nrows * sizeof(int))){
		LOG(ERROR) << "Error in reading nnz values from file!" << endl;
		return false;
	}

	// calculate how many number of entries we'd have to skip to get to the
	// region of file that is interesting locally
	long long begin_skip = std::accumulate(total_nnz_rows, total_nnz_rows + min_row, 0LL);
	local_num_nonzero = std::accumulate(total_nnz_rows + min_row, total_nnz_rows + max_row, 0LL);
	long long end_skip = total_nnz - local_num_nonzero - begin_skip;

	// Skip over the begin_nnz number of column indices in the file
	data_file.seekg(begin_skip * sizeof(int), std::ios_base::cur);

	//LOG(INFO) << "read column indices" << endl;

	int* col_idx = sallocator<int>().allocate(local_num_nonzero);
	if(!data_file.read(reinterpret_cast<char*>(col_idx), local_num_nonzero * sizeof(int))){
		LOG(ERROR) << "Error in reading column indices from file!" << endl;
		return false;
	}

	// Skip over remaining nnz and the beginning of data as well
	data_file.seekg(end_skip * sizeof(int) + begin_skip * sizeof(double), std::ios_base::cur);

	//LOG(INFO) << "read values" << endl;

	double* col_val = sallocator<double>().allocate(local_num_nonzero);
	if(!data_file.read(reinterpret_cast<char*>(col_val), local_num_nonzero * sizeof(double))){
		LOG(ERROR) << "Error in reading matrix values from file" << endl;
		exit(1);
	}

	data_file.close();

	// Now convert everything to column major format
	//LOG(INFO) << "form column-wise data structure" << endl;

	// First create vector of vectors
	vector<vector<int>, sallocator<int> > row_idx_vec(ncols);
	vector<vector<double>, sallocator<int> > row_val_vec(ncols);
	int* col_idx_ptr = col_idx;
	double* val_ptr = col_val;

	for(int i = min_row; i < max_row; i++){
		for(int j = 0; j < total_nnz_rows[i]; j++){
			int my_col_idx = *col_idx_ptr;
			double my_val = *val_ptr;
			// use relative indices for rows
			row_idx_vec[my_col_idx].push_back(i - min_row);
			row_val_vec[my_col_idx].push_back(static_cast<double>(my_val));
			col_idx_ptr++;
			val_ptr++;
		}
	}

	// Free up some space
	sallocator<int>().deallocate(col_idx, local_num_nonzero);
	sallocator<double>().deallocate(col_val, local_num_nonzero);
	sallocator<int>().deallocate(total_nnz_rows, nrows);

	//LOG(INFO) << "form CSC" << endl;

	// Now convert everything into CSC format

	col_offset.resize(ncols + 1);
	row_idx.resize(local_num_nonzero);
	row_val.resize(local_num_nonzero);

	int offset = 0;
	col_offset[0] = 0;

	for(int i = 0; i < ncols; i++){
		copy(row_idx_vec[i].begin(), row_idx_vec[i].end(), row_idx.begin() + offset);
		copy(row_val_vec[i].begin(), row_val_vec[i].end(), row_val.begin() + offset);
		offset += row_idx_vec[i].size();
		col_offset[i + 1] = offset;
	}

	row_idx_vec.clear();
	row_val_vec.clear();

	return true;

}

// -------- NomadBody --------

int NomadBody::get_num_cols(const std::string& path){

	const string train_filename = path + "/train.dat";

	// read number of columns from the data file
	int global_num_cols;
	{
		ifstream data_file(train_filename, ios::in | ios::binary);
		int header[4];

		if(!data_file.read(reinterpret_cast<char*>(header), 4 * sizeof(int))){
			LOG(ERROR) << "Error in reading ID from file" << endl;
			exit(11);
		}

		global_num_cols = header[2];

		data_file.close();
	}

	return global_num_cols;

}

bool NomadBody::load_train(const std::string& path,
	int part_index, int num_parts, Data& d
){

	const string train_filename = path + "/train.dat";
	return read_data(train_filename, part_index, num_parts,
		d.min_row_index, d.local_num_rows, d.local_num_nonzero,
		d.col_offset, d.row_idx, d.row_val,
		d.num_rows, d.num_cols, d.num_nonzero);

}

bool NomadBody::load_test(const std::string& path,
	int part_index, int num_parts, Data& d
){

	const string test_filename = path + "/test.dat";
	return read_data(test_filename, part_index, num_parts,
		d.min_row_index, d.local_num_rows, d.local_num_nonzero,
		d.col_offset, d.row_idx, d.row_val,
		d.num_rows, d.num_cols, d.num_nonzero);

}
