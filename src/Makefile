#
# this file was created by a computer. trust it.
#

# compiler tools
XILINX_VIVADO_HLS ?= $(XILINX_SDX)/Vivado_HLS

SDX_CXX ?= $(XILINX_SDX)/bin/xcpp
XOCC ?= $(XILINX_SDX)/bin/xocc
EMCONFIGUTIL = $(XILINX_SDX)/bin/emconfigutil --od .
RM = rm -f
RMDIR = rm -rf

SDX_PLATFORM = xilinx_aws-vu9p-f1-04261818_dynamic_5_0

MPI_CFLAGS := $(shell mpicc -showme:compiler)
MPI_LDFLAGS := $(shell mpicc -showme:linker)

# host compiler global settings
CXXFLAGS += -DSDX_PLATFORM=$(SDX_PLATFORM) -D__USE_XOPEN2K8 -I/opt/tools/xilinx/SDx/2018.2/runtime/include/1_2/ -I/opt/tools/xilinx/Vivado/2018.2/include/ -O0 -g -Wall -c -fmessage-length=0 -std=c++14 $(MPI_CFLAGS) -DSSSP -DSKIP_BFS
LDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++ -L/opt/tools/xilinx/SDx/2018.2/runtime/lib/x86_64 $(MPI_LDFLAGS)

# kernel compiler global settings
XOCC_OPTS = -t sw_emu --platform $(SDX_PLATFORM) --save-temps  --report system 

#
# OpenCL kernel files
#

BINARY_CONTAINERS += binary_container_1.xclbin

BUILD_SUBDIRS += binary_container_1
BINARY_CONTAINER_1_OBJS += binary_container_1/run_sssp.xo
ALL_KERNEL_OBJS += binary_container_1/run_sssp.xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

#
# host files
#

HOST_OBJECTS += src/main.o

HOST_EXE = sssp.exe

BUILD_SUBDIRS += src/

EMCONFIG_FILE = emconfig.json

#
# primary build targets
#

.PHONY: all clean
all: $(BINARY_CONTAINERS) $(HOST_EXE) $(EMCONFIG_FILE)

.NOTPARALLEL: clean

clean:
	-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES)  $(EMCONFIG_FILE) $(HOST_EXE) $(HOST_OBJECTS)
	-$(RM) *.xclbin.sh
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) _xocc*
	-$(RMDIR) .Xil

.PHONY: incremental
incremental: all

include makeemconfig.mk

nothing:

#
# binary container: binary_container_1.xclbin
#

binary_container_1/run_sssp.xo: ../src/sssp_reference.c /usr/include/stdio.h
	@mkdir -p $(@D)
	-@$(RM) $@
	$(XOCC) $(XOCC_OPTS) -c -k run_sssp -g  --nk run_sssp:1 --messageDb $(subst .xo,.mdb,$@) -I"$(<D)" --xp misc:solution_name=run_sssp --temp_dir binary_container_1 --report_dir binary_container_1/reports --log_dir binary_container_1/logs -o"$@" "$<"

binary_container_1.xclbin: $(BINARY_CONTAINER_1_OBJS)
	-@echo $(XOCC) $(XOCC_OPTS) -l --nk run_sssp:1 -g --messageDb $(subst .xclbin,.mdb,$@) --xp misc:solution_name=link --temp_dir binary_container_1 --report_dir binary_container_1/reports --log_dir binary_container_1/logs --remote_ip_cache /curr/jaywang/temp/karl/ip_cache -o"$@" $(+) > binary_container_1.xclbin.sh
	$(XOCC) $(XOCC_OPTS) -l --nk run_sssp:1 -g --messageDb $(subst .xclbin,.mdb,$@) --xp misc:solution_name=link --temp_dir binary_container_1 --report_dir binary_container_1/reports --log_dir binary_container_1/logs --remote_ip_cache /curr/jaywang/temp/karl/ip_cache -o"$@" $(+)

#
# host rules
#

src/%.o: ../src/%.c
	@mkdir -p $(@D)
	$(SDX_CXX) $(CXXFLAGS) -DSDX_PLATFORM=$(SDX_PLATFORM) -D__USE_XOPEN2K8 -I/opt/tools/xilinx/SDx/2018.2/runtime/include/1_2/ -I/opt/tools/xilinx/Vivado/2018.2/include/ -O0 -g -Wall -c -fmessage-length=0 -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(SDX_CXX) -o "$@" $(+) $(LDFLAGS) -lxilinxopencl -lpthread -lrt -lstdc++ -L/opt/tools/xilinx/SDx/2018.2/runtime/lib/x86_64 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/lib/csim -L $(XILINX_VIVADO)/lnx64/lib/csim -lhlsmc++-GCC46 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -L $(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -L $(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -lIp_xfft_v9_1_bitacc_cmodel -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -L $(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -lgmp -lIp_fir_compiler_v7_2_bitacc_cmodel -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/dds_v6_0 -L $(XILINX_VIVADO)/lnx64/tools/dds_v6_0 -lIp_dds_compiler_v6_0_bitacc_cmodel 

#
# emulation configuration file
#

$(EMCONFIG_FILE): makeemconfig.mk
	$(EMCONFIGUTIL) --nd $(NUMBER_OF_DEVICES)  --platform $(SDX_PLATFORM)
	-@$(RMDIR) TempConfig .Xil/configutil*

