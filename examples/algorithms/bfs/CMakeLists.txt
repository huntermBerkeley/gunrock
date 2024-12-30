# begin /* Set the application name. */
set(APPLICATION_NAME bfs)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${APPLICATION_NAME})

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
)

target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message(STATUS "Example Added: ${APPLICATION_NAME}")
# end /* Add CUDA executables */



# begin /* Set the application name. */
set(ALT_APPLICATION_NAME graph_bench)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${ALT_APPLICATION_NAME})

set(SOURCE_LIST 
    ${ALT_APPLICATION_NAME}.cu
)

target_sources(${ALT_APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${ALT_APPLICATION_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${ALT_APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message(STATUS "Example Added: ${ALT_APPLICATION_NAME}")
# end /* Add CUDA executables */



# begin /* Set the application name. */
set(SSSP_APP_NAME sssp_cache)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${SSSP_APP_NAME})

set(SOURCE_LIST 
    ${SSSP_APP_NAME}.cu
)

target_sources(${SSSP_APP_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${SSSP_APP_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${SSSP_APP_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message(STATUS "Example Added: ${SSSP_APP_NAME}")
# end /* Add CUDA executables */