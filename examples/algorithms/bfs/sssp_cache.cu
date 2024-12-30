#include <gunrock/algorithms/sssp.hxx>
#include <gunrock/algorithms/dawn.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/framework/benchmark.hxx>

#include <caching/pointers/set_cache.cuh>

#include "bfs_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;


bool fileExists(const std::string& filename) {
  std::ifstream file(filename);
  bool exists = file.good();
  return exists;
}

template<memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
format::csr_t<space, index_t, offset_t, value_t> load_graph(gunrock::io::cli::parameters_t params, format::coo_no_vector<space, index_t, offset_t, value_t> coo){

  format::csr_t<space, index_t, offset_t, value_t> csr;

  std::string filename = params.filename;

  if (fileExists(filename+".csr")){

    printf("Reading CSR binary\n");

    csr.read_in_csr(filename+".csr");

  } else {

    printf("converting COO\n");

    csr.from_coo_large(coo);

    std::string binary_name = params.filename+".csr";

    printf("Writing csr file %s\n", binary_name.c_str());

    csr.write_out_csr(binary_name);

  }

  return csr;

}


template<memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
format::csr_t<space, index_t, offset_t, value_t> load_csr(gunrock::io::cli::parameters_t params){

  format::csr_t<space, index_t, offset_t, value_t> csr;

  std::string filename = params.filename;
  
  csr.read_in_csr(filename+".csr");

  return csr;

}


bool has_csr(gunrock::io::cli::parameters_t params){

  return fileExists(params.filename+".csr");

}

void test_cache_sssp(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = uint64_t;
  using edge_t = uint64_t;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;

  // auto coo_t = 
  //     format::coo_no_vector<memory_space_t::host, vertex_t, edge_t, weight_t>;

  std::string DEFAULT_SSSP_ALGORITHMS =
      "DAWN";  // Using 'Breadth First Search' here will call the original BFS
  // --
  // IO


      if (has_csr){

      }

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        DEFAULT_SSSP_ALGORITHMS);


  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;



  gunrock::graph::graph_properties_t properties;

  csr_t csr;

  if (has_csr(params)){


    properties = mm.load_properties(params.filename);

    csr = load_csr<memory_space_t::host, vertex_t, edge_t, weight_t>(params);


  } else {

    auto [ext_properties, coo] = mm.load_large(params.filename);

    properties = ext_properties;
    csr = load_graph<memory_space_t::host, vertex_t, edge_t, weight_t>(params, coo);

  }


  // auto csr_properties = mm.load_properties(params.filename);
  // auto [properties, coo] = mm.load_large(params.filename);


  // auto [properties, coo] = mm.load_large(params.filename);

  // csr_t csr = load_graph<memory_space_t::host, vertex_t, edge_t, weight_t>(params, coo);

  // if (params.filename) {

  //   printf("Reading binary\n");
  //   csr.read_binary(params.filename);
  // } else {

  //   printf("Reading from COO\n");

  //   csr.from_coo_large(coo);

  //   std::string binary_name = params.filename+".csr";

  //   printf("Writing csr file %s\n", binary_name.c_str());

  //   csr.write_binary(binary_name);
    
  // }

  // --
  // Build graph

  printf("CSR BUILT\n");

  auto G = graph::build<memory_space_t::host>(properties, csr);


  printf("GRAPH BUILT %lu vert\n", G.get_number_of_vertices());
  // --
  // Params and memory allocation

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  // thrust::device_vector<vertex_t> distances(n_vertices);
  // thrust::device_vector<vertex_t> predecessors(n_vertices);

  // thrust::host_vector<vertex_t> distances(n_vertices);
  // thrust::host_vector<vertex_t> predecessors(n_vertices);


  weight_t * distances;

  cudaMallocHost((void **)&distances, sizeof(weight_t)*n_vertices);

  vertex_t * predecessors;

  cudaMallocHost((void **)&predecessors, sizeof(vertex_t)*n_vertices);

  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> vertices_visited(1);

  printf("Thrust memory alloced\n");

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // Run problem


  printf("Mem allocated, starting execution\n");

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;


  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {

    vertex_t source = source_vect[i];
    benchmark::INIT_BENCH();
    if (DEFAULT_SSSP_ALGORITHMS == "DAWN")
      run_times.push_back(gunrock::dawn_sssp::run(G, source,
                                                  distances,
                                                  predecessors));
    else
      run_times.push_back(gunrock::sssp::run(G, source,
                                             distances,
                                             predecessors));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    if (DEFAULT_SSSP_ALGORITHMS == "DAWN")
      gunrock::util::stats::export_performance_stats(
          benchmark_metrics, n_edges, n_vertices, run_times, "dawn_bfs",
          params.filename, "market", params.json_dir, params.json_file,
          source_vect, tag_vect, num_arguments, argument_array);
    else
      gunrock::util::stats::export_performance_stats(
          benchmark_metrics, n_edges, n_vertices, run_times, "bfs",
          params.filename, "market", params.json_dir, params.json_file,
          source_vect, tag_vect, num_arguments, argument_array);
  }

  //output.
  //print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // CPU Run

  // if (params.validate) {
  //   thrust::host_vector<vertex_t> h_distances(n_vertices);
  //   thrust::host_vector<vertex_t> h_predecessors(n_vertices);

  //   // Validate with last source in source vector
  //   float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
  //       csr, source_vect.back(), h_distances.data(), h_predecessors.data());

  //   int n_errors =
  //       util::compare(distances, h_distances.data(), n_vertices);
  //   print::head(h_distances, 40, "CPU Distances");

  //   std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  //   std::cout << "Number of errors : " << n_errors << std::endl;
  // }
}

int main(int argc, char** argv) {
  //test_bfs_dev(argc, argv);
  test_cache_sssp(argc, argv);
}
