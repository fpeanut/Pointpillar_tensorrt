cmake_minimum_required(VERSION 3.10)
project(ApolloPro LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
# 获取上一级目录
# get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# message("camke path: ${PARENT_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${PARENT_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
include_directories(
    .
    src
)
# include_directories(
#     src
#     # pointpillars/scatterND/
# )
file(GLOB SRC
    src/*.cu 
    src/*.cc
)

#编译可执行文件
add_executable(ApolloPP demo.cc ${SRC})
target_include_directories(ApolloPP PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(ApolloPP PRIVATE  -Wl,--no-as-needed ${TensorRT_LIBRARIES} yaml-cpp ${CUDA_LIBRARIES})  
set_target_properties(ApolloPP PROPERTIES CUDA_ARCHITECTURES "61;70;75;86")#这个cuda必须要有

#genrate anchor 可执行文件
add_executable(generate_anchor gen_anchor.cc)

