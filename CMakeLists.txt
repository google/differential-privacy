cmake_minimum_required(VERSION 3.16)
project(DifferentialPrivacyAlgorithms LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch protobuf
include(FetchContent)
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.21.12
  # For older CMake versions, you might need SOURCE_DIR for protobuf
  SOURCE_SUBDIR  cmake
  FIND_PACKAGE_ARGS NAMES protobuf
)

# CMake variables to control the protobuf build
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE) # Usually good to disable tests of dependencies

FetchContent_MakeAvailable(protobuf)
include(FindProtobuf)
find_package(protobuf CONFIG REQUIRED)

# Include directories for the project
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(PROTO_FILES
  proto/summary.proto
  proto/data.proto
  proto/confidence-interval.proto
  proto/numerical-mechanism.proto
)
message(STATUS ${PROTO_FILES})

# Generate C++ sources and headers from .proto files
set(Protobuf_IMPORT_DIRS "${protobuf_SOURCE_DIR}/src" .)
add_library(dp_protos)
protobuf_generate(TARGET dp_protos
                  PROTOS ${PROTO_FILES}
                  IMPORT_DIRS ${Protobuf_IMPORT_DIRS}
                  PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})


# Add a library for the generated protobuf files
target_include_directories(dp_protos PUBLIC ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/.." ..)
target_link_libraries(dp_protos
    PRIVATE
        protobuf::libprotobuf
)

# Add cc subdirectories
add_subdirectory(cc)