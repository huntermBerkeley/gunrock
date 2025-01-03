#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/memory.hxx>
#include <gunrock/util/load_store.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/algorithms/search/binary_search.hxx>
#include <gunrock/formats/formats.hxx>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include <caching/pointers/set_cache.cuh>
#include <caching/pointers/host_device_allocator.cuh>

namespace gunrock {
namespace graph {

struct empty_csr_t {};

using namespace memory;

// XXX: The ideal thing to do here is to inherit
// base class with virtual keyword specifier, therefore
// public virtual graph_base_t<...> {}, but according to
// cuda's programming guide, that is not allowerd.
// From my tests (see smart_struct) results in an illegal
// memory error. Another important thing to note is that
// virtual functions should also have undefined behavior,
// but they seem to work.
template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t, typename cache_type=caching::set_cache<64, 16>>

// template <memory_space_t space,
//           typename vertex_t,
//           typename edge_t,
//           typename weight_t, typename cache_type=caching::device_allocator>
class graph_csr_t {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pair_type = vertex_pair_t<vertex_type>;

  using offset_type = typename cache_type::pointer_type<edge_type>;

  using index_type = typename cache_type::pointer_type<vertex_type>;
  using value_type = typename cache_type::pointer_type<weight_type>;


 public:
  __host__ __device__ graph_csr_t()
      : offsets(nullptr, nullptr), indices(nullptr, nullptr), values(nullptr, nullptr) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/

  // Copy constructor
  // graph_csr_t(const graph_csr_t& rhs) = delete;
  // Copy assignment
  // graph_csr_t& operator=(const graph_csr_t& rhs) = delete;

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const {
    return (get_starting_edge(v + 1) - get_starting_edge(v));
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const {
    auto keys = get_row_offsets();
    auto key = e;

    // returns `it` such that everything to the left is <= e.
    // This will be one element to the right of the node id.
    auto it = thrust::lower_bound(
        thrust::seq, thrust::counting_iterator<edge_t>(0),
        thrust::counting_iterator<edge_t>(this->number_of_vertices), key,
        [keys] __host__ __device__(const edge_t& pivot, const edge_t& key) {
          return keys[pivot] <= key;
        });

    return (*it) - 1;
  }

  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const {


    return thread::load<index_type, vertex_type>(indices, e);
    //return thread::load(&indices[e]);
  }

  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const {

    return thread::load<offset_type, edge_type>(offsets, v);
    //return thread::load(&offsets[v]);
  }

  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(const edge_type& e) const {
    return {get_source_vertex(e), get_destination_vertex(e)};
  }

  // TODO: this uses 1-based indexing while other views use 0-based indexing
  __host__ __device__ __forceinline__ edge_type
  get_edge(const vertex_type& source, const vertex_type& destination) const {
    return (edge_type)search::binary::execute(
        get_column_indices(), destination, get_starting_edge(source),
        get_starting_edge(source + 1) - 1);
  }

  /**
   * @brief Count the number of vertices belonging to the set intersection
   * between the source and destination vertices adjacency lists. Executes a
   * function on each intersection. This function does not handle self-loops.
   *
   * @param source Index of the source vertex
   * @param destination Index of the destination
   * @param on_intersection Lambda function executed at each intersection
   * @return Number of shared vertices between source and destination
   */
  template <typename operator_type>
  __host__ __device__ __forceinline__ vertex_type
  get_intersection_count(const vertex_type& source,
                         const vertex_type& destination,
                         operator_type on_intersection) const {
    vertex_type intersection_count = 0;

    auto intersection_source = source;
    auto intersection_destination = destination;

    auto source_neighbors_count = get_number_of_neighbors(source);
    auto destination_neighbors_count = get_number_of_neighbors(destination);

    if (source_neighbors_count == 0 || destination_neighbors_count == 0) {
      return 0;
    }
    if (source_neighbors_count > destination_neighbors_count) {
      thrust::swap(intersection_source, intersection_destination);
      thrust::swap(source_neighbors_count, destination_neighbors_count);
    }

    auto source_offset = offsets[intersection_source];
    auto destination_offset = offsets[intersection_destination];

    auto source_edges_iter = indices + source_offset;
    auto destination_edges_iter = indices + destination_offset;

    auto needle = *destination_edges_iter;

    auto source_search_start = thrust::distance(
        source_edges_iter,
        thrust::lower_bound(thrust::seq, source_edges_iter,
                            source_edges_iter + source_neighbors_count,
                            needle));

    if (source_search_start == source_neighbors_count) {
      return 0;
    }
    edge_type destination_search_start = 0;

    while (source_search_start < source_neighbors_count &&
           destination_search_start < destination_neighbors_count) {
      auto cur_edge_src = source_edges_iter[source_search_start];
      auto cur_edge_dst = destination_edges_iter[destination_search_start];
      if (cur_edge_src == cur_edge_dst) {
        intersection_count++;
        source_search_start++;
        destination_search_start++;
        on_intersection(cur_edge_src);
      } else if (cur_edge_src > cur_edge_dst) {
        destination_search_start++;
      } else {
        source_search_start++;
      }
    }

    return intersection_count;
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const {

    return thread::load<value_type, weight_type>(values, e);
    //return thread::load(&values[e]);
  }

  // Representation specific functions
  // ...
  __host__ __device__ __forceinline__ auto get_row_offsets() const {
    return offsets;
  }

  __host__ __device__ __forceinline__ auto get_column_indices() const {
    return indices;
  }

  __host__ __device__ __forceinline__ auto get_nonzero_values() const {
    return values;
  }

  // Graph type (inherited from this class) has equivalents of this in graph
  // terminology (vertices and edges). Also include these for linear algebra
  // terminology
  __host__ __device__ __forceinline__ auto get_number_of_rows() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_columns() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_nonzeros() const {
    return number_of_edges;
  }

  __host__ __device__ __forceinline__ auto get_number_of_vertices() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_edges() const {
    return number_of_edges;
  }

 protected:
  __host__ void set(
      gunrock::format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
    this->number_of_vertices = csr.number_of_rows;
    this->number_of_edges = csr.number_of_nonzeros;
    // Set raw pointers

    printf("Protected set is called\n");

    cache_type * cache = cache_type::generate_on_device(1024ULL*14ULL*1024ULL);


  
    // index_t number_of_rows;
    // index_t number_of_columns;
    // offset_t number_of_nonzeros;

    //vertex, edge, weight
    //index offset value

    //row offsets - offset_t

    //column indices - index_t

    //nonzero_values - value_t

    printf("Init offsets with %llu\n rows.\n", csr.number_of_nonzeros);

    edge_type * host_edges = gallatin::utils::get_host_version<edge_type>(csr.number_of_rows+1);

    edge_type * thrust_edges = thrust::raw_pointer_cast(csr.row_offsets.data());

    cudaMemcpy(host_edges, thrust_edges, sizeof(edge_type)*(csr.number_of_rows+1), cudaMemcpyHostToHost);

    vertex_type * host_indices = gallatin::utils::get_host_version<vertex_type>(csr.number_of_nonzeros);

    cudaMemcpy(host_indices, thrust::raw_pointer_cast(csr.column_indices.data()), sizeof(vertex_type)*csr.number_of_nonzeros, cudaMemcpyHostToHost);

    weight_type * host_values = gallatin::utils::get_host_version<weight_type>(csr.number_of_nonzeros);

    cudaMemcpy(host_values, thrust::raw_pointer_cast(csr.nonzero_values.data()), sizeof(weight_type)*csr.number_of_nonzeros, cudaMemcpyHostToHost);

    offsets = offset_type(host_edges, cache);
    indices = index_type(host_indices, cache);
    values = value_type(host_values, cache) ;
  }

 private:
  // Underlying data storage
  vertex_type number_of_vertices;
  edge_type number_of_edges;


  offset_type offsets;
  index_type indices;
  value_type values;

};  // struct graph_csr_t

}  // namespace graph
}  // namespace gunrock