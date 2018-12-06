COMMON_REPO := /home/centos/src/project_data/aws-fpga/SDAccel/examples/xilinx

# wide Memory Access Application
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk
include $(COMMON_REPO)/utility/rules.mk

# dot product Host Application
main_SRCS=./src/main.c $(xcl2_SRCS)
main_HDRS=$(xcl2_HDRS)
main_CXXFLAGS=-I./src/ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS)
main_LDFLAGS=$(opencl_LDFLAGS)

MPI_CFLAGS := $(shell mpicc -showme:compiler)
MPI_LDFLAGS := $(shell mpicc -showme:linker)

CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION -I../aml $(MPI_CFLAGS) $(main_CXXFLAGS)
LDFLAGS = -lpthread $(MPI_LDFLAGS) $(main_LDFLAGS)

CC = gcc

TARGETS := hw
TARGET := $(TARGETS)
DEVICES := xilinx_vcu1525_dynamic
DEVICE := $(DEVICES)
XCLBIN := ./xclbin
DSA := $(call device2sandsa, $(DEVICE))

all: graph500_sssp

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = main.c utils.c validate.c ../aml/aml.c
HEADERS = common.h csr_reference.h bitmap_reference.h

graph500_sssp: bfs_reference.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c sssp_reference.c
	$(CC) $(CFLAGS) -DSSSP -DSKIP_BFS $(LDFLAGS) -o graph500_sssp bfs_reference.c sssp_reference.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

# check
check_EXE=graph500_sssp 18

CHECKS=check

clean:
	-rm -f graph500_*
