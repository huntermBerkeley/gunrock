#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>

namespace gunrock {
namespace format {

using namespace memory;

/**
 * @brief Coordinate (COO) format.
 *
 * @tparam index_t
 * @tparam nz_size_t
 * @tparam value_t
 */
template <memory_space_t space,
          typename index_t,
          typename nz_size_t,
          typename value_t>
struct coo_no_vector {
  index_t number_of_rows;
  index_t number_of_columns;
  nz_size_t number_of_nonzeros;

  index_t * row_indices;     // I
  index_t * column_indices;  // J
  value_t * nonzero_values;  // V

  coo_no_vector()
      : number_of_rows(0),
        number_of_columns(0),
        number_of_nonzeros(0),
        row_indices(),
        column_indices(),
        nonzero_values() {}

  coo_no_vector(index_t r, index_t c, nz_size_t nnz)
      : number_of_rows(r),
        number_of_columns(c),
        number_of_nonzeros(nnz)
        {


          index_t * _I;
          index_t* _J;
          value_t * _V;



          cudaMallocHost((void **)&_I, sizeof(index_t)*nnz);
          cudaMallocHost((void **)&_J, sizeof(index_t)*nnz);
          cudaMallocHost((void **)&_V, sizeof(value_t)*nnz);

          row_indices = _I;
          column_indices = _J;
          nonzero_values = _V;


        }

  ~coo_no_vector() {

    // if (row_indices != nullptr){
    //   cudaFreeHost(row_indices);
    // }

    // if (column_indices != nullptr){
    //   cudaFreeHost(column_indices);
    // }

    // if (nonzero_values != nullptr){
    //   cudaFreeHost(nonzero_values);
    // }
  }

  /**
   * @brief Convert CSR format into COO
   * Format.
   *
   * @tparam index_t
   * @tparam index_t
   * @tparam value_t
   * @param csr
   * @return coo_no_vector<space, index_t, index_t, value_t>&
   */
  // TODO: fix index_t -> offset_t
  // coo_no_vector<space, index_t, index_t, value_t> from_csr(
  //     const csr_t<memory_space_t::host, index_t, index_t, value_t>& csr) {
  //   number_of_rows = csr.number_of_rows;
  //   number_of_columns = csr.number_of_columns;
  //   number_of_nonzeros = csr.number_of_nonzeros;


  //   vertex_t * _I;
  //   vertex_t* _J;
  //   weight_t * _V;



  //   cudaMallocHost((void **)&_I, sizeof(vertex_t)*number_of_nonzeros);
  //   cudaMallocHost((void **)&_J, sizeof(vertex_t)*number_of_nonzeros);
  //   cudaMallocHost((void **)&_V, sizeof(weight_t)*number_of_nonzeros);

  //   row_indices = _I;
  //   column_indices = _J;
  //   nonzero_values = _V;

  //   row_indices.resize(number_of_nonzeros);
  //   column_indices.resize(number_of_nonzeros);
  //   nonzero_values.resize(number_of_nonzeros);

  //   // Convert offsets to indices
  //   gunrock::graph::convert::offsets_to_indices<memory_space_t::host>(
  //       memory::raw_pointer_cast(csr.row_offsets.data()),
  //       csr.number_of_rows + 1, memory::raw_pointer_cast(row_indices.data()),
  //       number_of_nonzeros);

  //   column_indices = csr.column_indices;
  //   nonzero_values = csr.nonzero_values;

  //   return *this;  // COO representation
  // }

};  // struct coo_no_vector

}  // namespace format
}  // namespace gunrock