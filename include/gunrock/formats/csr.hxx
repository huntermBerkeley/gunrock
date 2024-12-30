#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/error.hxx>

#include <gunrock/container/vector.hxx>
#include <gunrock/formats/formats.hxx>

#include <thrust/transform.h>

#include <caching/pointers/progress.cuh>

#include <iostream>
#include <fstream>

namespace gunrock {
namespace format {

using namespace memory;

/**
 * @brief Compressed Sparse Row (CSR) format.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
struct csr_t {
  using index_type = index_t;
  using offset_type = offset_t;
  using value_type = value_t;

  index_t number_of_rows;
  index_t number_of_columns;
  offset_t number_of_nonzeros;

  vector_t<offset_t, space> row_offsets;    // Ap
  vector_t<index_t, space> column_indices;  // Aj
  vector_t<value_t, space> nonzero_values;  // Ax

  csr_t()
      : number_of_rows(0),
        number_of_columns(0),
        number_of_nonzeros(0),
        row_offsets(),
        column_indices(),
        nonzero_values() {}

  csr_t(index_t r, index_t c, offset_t nnz)
      : number_of_rows(r),
        number_of_columns(c),
        number_of_nonzeros(nnz),
        row_offsets(r + 1),
        column_indices(nnz),
        nonzero_values(nnz) {}

  ~csr_t() {}

  /**
   * @brief Copy constructor.
   * @param rhs
   */
  template <typename _csr_t>
  csr_t(const _csr_t& rhs)
      : number_of_rows(rhs.number_of_rows),
        number_of_columns(rhs.number_of_columns),
        number_of_nonzeros(rhs.number_of_nonzeros),
        row_offsets(rhs.row_offsets),
        column_indices(rhs.column_indices),
        nonzero_values(rhs.nonzero_values) {}

  /**
   * @brief Convert a Coordinate Sparse Format into Compressed Sparse Row
   * Format.
   *
   * @tparam index_t
   * @tparam offset_t
   * @tparam value_t
   * @param coo
   * @return csr_t<space, index_t, offset_t, value_t>&
   */
  csr_t<space, index_t, offset_t, value_t> from_coo(
      const coo_t<memory_space_t::host, index_t, offset_t, value_t>& coo) {

    printf("Reading from COO\n");
    number_of_rows = coo.number_of_rows;
    number_of_columns = coo.number_of_columns;
    number_of_nonzeros = coo.number_of_nonzeros;

    // Allocate space for vectors
    vector_t<offset_t, memory_space_t::host> Ap;
    vector_t<index_t, memory_space_t::host> Aj;
    vector_t<value_t, memory_space_t::host> Ax;

    // offset_t* Ap;
    // index_t* Aj;
    // value_t* Ax;

    Ap.resize(number_of_rows + 1);
    Aj.resize(number_of_nonzeros);
    Ax.resize(number_of_nonzeros);

    printf("CSR Allocated\n");
    // Ap = _Ap.data();
    // Aj = _Aj.data();
    // Ax = _Ax.data();

    // compute number of non-zero entries per row of A.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      ++Ap[coo.row_indices[n]];
    }

    // cumulative sum the nnz per row to get row_offsets[].
    for (index_t i = 0, sum = 0; i < number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = sum;
      sum += temp;
    }
    Ap[number_of_rows] = number_of_nonzeros;

    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      index_t row = coo.row_indices[n];
      index_t dest = Ap[row];

      Aj[dest] = coo.column_indices[n];
      Ax[dest] = coo.nonzero_values[n];

      ++Ap[row];
    }

    for (index_t i = 0, last = 0; i <= number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = last;
      last = temp;
    }

    row_offsets = Ap;
    column_indices = Aj;
    nonzero_values = Ax;

    printf("Done\n");

    return *this;  // CSR representation (with possible duplicates)
  }

  csr_t<space, index_t, offset_t, value_t> from_coo_large(
      const coo_no_vector<memory_space_t::host, index_t, offset_t, value_t>& coo) {

    printf("Reading from COO\n");
    number_of_rows = coo.number_of_rows;
    number_of_columns = coo.number_of_columns;
    number_of_nonzeros = coo.number_of_nonzeros;

    // Allocate space for vectors
    vector_t<offset_t, memory_space_t::host> Ap;
    vector_t<index_t, memory_space_t::host> Aj;
    vector_t<value_t, memory_space_t::host> Ax;

    // offset_t* Ap;
    // index_t* Aj;
    // value_t* Ax;

    Ap.resize(number_of_rows + 1);
    Aj.resize(number_of_nonzeros);
    Ax.resize(number_of_nonzeros);

    // Ap = _Ap.data();
    // Aj = _Aj.data();
    // Ax = _Ax.data();

    // compute number of non-zero entries per row of A.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      ++Ap[coo.row_indices[n]];
    }

    printf("CSR [1/3]\n");

    // cumulative sum the nnz per row to get row_offsets[].
    for (index_t i = 0, sum = 0; i < number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = sum;
      sum += temp;
      display_progress(i, number_of_rows, .01);
    }

    end_bar(number_of_rows);
    Ap[number_of_rows] = number_of_nonzeros;

    printf("CSR [2/3]\n");
    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      index_t row = coo.row_indices[n];
      index_t dest = Ap[row];

      Aj[dest] = coo.column_indices[n];
      Ax[dest] = coo.nonzero_values[n];

      ++Ap[row];
      display_progress(n, number_of_nonzeros, .01);
    }
    end_bar(number_of_nonzeros);

    printf("CSR [3/3]\n");
    for (index_t i = 0, last = 0; i <= number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = last;
      last = temp;

      display_progress(i, number_of_rows, .01);
    }
    end_bar(number_of_rows);

    row_offsets = Ap;
    column_indices = Aj;
    nonzero_values = Ax;

    printf("Done with CSR\n");

    return *this;  // CSR representation (with possible duplicates)
  }

  void read_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "rb");

    // Read metadata
    error::throw_if_exception(
        fread(&number_of_rows, sizeof(index_t), 1, file) != 0);
    error::throw_if_exception(
        fread(&number_of_columns, sizeof(index_t), 1, file) != 0);
    error::throw_if_exception(
        fread(&number_of_nonzeros, sizeof(offset_t), 1, file) != 0);

    row_offsets.resize(number_of_rows + 1);
    column_indices.resize(number_of_nonzeros);
    nonzero_values.resize(number_of_nonzeros);

    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_row_offsets(number_of_rows + 1);
      thrust::host_vector<index_t> h_column_indices(number_of_nonzeros);
      thrust::host_vector<value_t> h_nonzero_values(number_of_nonzeros);

      error::throw_if_exception(
          fread(memory::raw_pointer_cast(h_row_offsets.data()),
                sizeof(offset_t), number_of_rows + 1, file) != 0);
      error::throw_if_exception(
          fread(memory::raw_pointer_cast(h_column_indices.data()),
                sizeof(index_t), number_of_nonzeros, file) != 0);
      error::throw_if_exception(
          fread(memory::raw_pointer_cast(h_nonzero_values.data()),
                sizeof(value_t), number_of_nonzeros, file) != 0);

      // Copy data from host to device
      row_offsets = h_row_offsets;
      column_indices = h_column_indices;
      nonzero_values = h_nonzero_values;

    } else {
      assert(space == memory_space_t::host);

      error::throw_if_exception(
          fread(memory::raw_pointer_cast(row_offsets.data()), sizeof(offset_t),
                number_of_rows + 1, file) != 0);
      error::throw_if_exception(
          fread(memory::raw_pointer_cast(column_indices.data()),
                sizeof(index_t), number_of_nonzeros, file) != 0);
      error::throw_if_exception(
          fread(memory::raw_pointer_cast(nonzero_values.data()),
                sizeof(value_t), number_of_nonzeros, file) != 0);
    }

    fclose(file);
  }

  void write_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "wb");

    // Write metadata
    fwrite(&number_of_rows, sizeof(index_t), 1, file);
    fwrite(&number_of_columns, sizeof(index_t), 1, file);
    fwrite(&number_of_nonzeros, sizeof(offset_t), 1, file);

    // Write data
    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_row_offsets(row_offsets);
      thrust::host_vector<index_t> h_column_indices(column_indices);
      thrust::host_vector<value_t> h_nonzero_values(nonzero_values);

      fwrite(memory::raw_pointer_cast(h_row_offsets.data()), sizeof(offset_t),
             number_of_rows + 1, file);
      fwrite(memory::raw_pointer_cast(h_column_indices.data()), sizeof(index_t),
             number_of_nonzeros, file);
      fwrite(memory::raw_pointer_cast(h_nonzero_values.data()), sizeof(value_t),
             number_of_nonzeros, file);
    } else {
      assert(space == memory_space_t::host);

      fwrite(memory::raw_pointer_cast(row_offsets.data()), sizeof(offset_t),
             number_of_rows + 1, file);
      fwrite(memory::raw_pointer_cast(column_indices.data()), sizeof(index_t),
             number_of_nonzeros, file);
      fwrite(memory::raw_pointer_cast(nonzero_values.data()), sizeof(value_t),
             number_of_nonzeros, file);
    }

    fclose(file);
  }

  template <typename T>
  void filewrite(std::ofstream & file, T data){
    file.write(reinterpret_cast<char*>(&data), sizeof(T));
  }

  template <typename T>
  void filewrite_array(std::ofstream & file, T * data, uint64_t n_items){
    file.write(reinterpret_cast<char*>(data), sizeof(T)*n_items);
  }

  template <typename T>
  void fileread(std::ifstream & file, T & data){
    file.read(reinterpret_cast<char*>(&data), sizeof(T));
  }

  template <typename T>
  void fileread_array(std::ifstream & file, T * data, uint64_t n_items){
    file.read(reinterpret_cast<char*>(data), sizeof(T)*n_items);
  }

  void write_out_csr(std::string filename){

    std::ofstream outfile(filename, std::ios::binary);

    printf("Writing out %lu rows %lu cols %lu nnz\n", number_of_rows, number_of_columns, number_of_nonzeros);
    printf("Datatype sizes, %lu %lu %lu\n", sizeof(number_of_rows), sizeof(number_of_columns), sizeof(number_of_nonzeros));


    filewrite(outfile, number_of_rows);
    filewrite(outfile, number_of_columns);
    filewrite(outfile, number_of_nonzeros);

    assert(space == memory_space_t::host);

    offset_t * output_rows = memory::raw_pointer_cast(row_offsets.data());

  // vector_t<index_t, space> column_indices;  // Aj
  // vector_t<value_t, space> nonzero_values;  // Ax

    printf("First and last: %lu %lu\n", output_rows[0], output_rows[number_of_rows]);

    filewrite_array(outfile, output_rows, number_of_rows+1);
    // for (uint64_t i=0; i < number_of_rows+1; i++){
    //   outfile << output_rows[i] << " ";
    // }

    index_t * output_cols = memory::raw_pointer_cast(column_indices.data());

    // for (uint64_t i=0; i < number_of_nonzeros; i++){
    //   outfile << output_cols[i] << " ";
    // }

    filewrite_array(outfile, output_cols, number_of_nonzeros);

    printf("First and last: %lu %lu\n", output_cols[0], output_cols[number_of_nonzeros-1]);

    value_t * output_vals = memory::raw_pointer_cast(nonzero_values.data());

    // for (uint64_t i = 0; i < number_of_nonzeros; i++){
    //   outfile << output_vals[i] << " ";
    // }

    filewrite_array(outfile, output_vals, number_of_nonzeros);

    printf("First and last: %f %f\n", output_vals[0], output_vals[number_of_nonzeros-1]);

  }

  void read_in_csr(std::string filename){
    std::ifstream infile(filename, std::ios::binary);

    fileread(infile, number_of_rows);
    fileread(infile, number_of_columns);
    fileread(infile, number_of_nonzeros);
    //infile >> number_of_rows >> number_of_columns >> number_of_nonzeros;

    printf("File with %lu rows %lu cols %lu nnz\n", number_of_rows, number_of_columns, number_of_nonzeros);
    //printf("Datatype sizes, %lu %lu %lu\n", sizeof(number_of_rows), sizeof(number_of_columns), sizeof(number_of_nonzeros));



    thrust::host_vector<offset_t> h_row_offsets(number_of_rows + 1);
    thrust::host_vector<index_t> h_column_indices(number_of_nonzeros);
    thrust::host_vector<value_t> h_nonzero_values(number_of_nonzeros);

    // for (uint64_t i = 0; i < number_of_rows; i++){
    //   infile >> h_row_offsets[i];
    // }

    offset_t * input_rows = memory::raw_pointer_cast(h_row_offsets.data());

    fileread_array(infile, input_rows, number_of_rows+1);

    printf("First and last: %lu %lu\n", h_row_offsets[0], h_row_offsets[number_of_rows]);

    // for (uint64_t i = 0; i < number_of_nonzeros; i++){
    //   infile >> h_column_indices[i];
    // }

    index_t * input_cols = memory::raw_pointer_cast(h_column_indices.data());

    fileread_array(infile, input_cols, number_of_nonzeros);

    printf("First and last: %lu %lu\n", h_column_indices[0], h_column_indices[number_of_nonzeros-1]);


    // for (uint64_t i = 0; i < number_of_nonzeros; i++){
    //   infile >> h_nonzero_values[i];
    // }

    value_t * input_vals = memory::raw_pointer_cast(h_nonzero_values.data());

    fileread_array(infile, input_vals, number_of_nonzeros);

    printf("First and last: %f %f\n", h_nonzero_values[0], h_nonzero_values[number_of_nonzeros-1]);

    row_offsets = h_row_offsets;
    column_indices = h_column_indices;
    nonzero_values = h_nonzero_values;

    printf("Done with read\n");

  }

};  // struct csr_t

}  // namespace format
}  // namespace gunrock