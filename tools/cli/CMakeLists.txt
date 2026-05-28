set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)
include_directories(${CMAKE_SOURCE_DIR}/examples/speculative)

# Phase 29F: propagate PRT_SIDECAR_PAGER_EXPERIMENTAL to CLI so llama_set_prt_flags()
# is compiled into the CLI (without this flag, the entire PRT init block is #ifdef'd out)
if(LLAMA_PRT_SIDECAR_PAGER)
    target_compile_options(${TARGET} PRIVATE -DPRT_SIDECAR_PAGER_EXPERIMENTAL)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
