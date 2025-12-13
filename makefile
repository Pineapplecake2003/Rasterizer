root_dir := $(PWD)
sim_dir := ./sim
src_dir := ./src
inc_dir := ./include
bld_dir := ./build
GOLDEN_PATH := ../golden/points.txt
FSDB_DEF :=
ifeq ($(FSDB),1)
FSDB_DEF := +FSDB
else ifeq ($(FSDB),2)
FSDB_DEF := +FSDB_ALL
endif

$(bld_dir):
	mkdir -p $(bld_dir)



rtl: | $(bld_dir)
	cd $(bld_dir) && \
	vcs -R -sverilog $(root_dir)/$(sim_dir)/Rasterizer_tb.sv \
	-debug_access+all -full64 \
	+incdir+$(root_dir)/$(src_dir)+$(root_dir)/$(src_dir)/AXI+$(root_dir)/$(inc_dir)+$(root_dir)/$(sim_dir) \
	$(FSDB_DEF) \
	+golden_path=$(GOLDEN_PATH) \
	+rdcycle=1 \
	+notimingcheck



gen_golden:
	./script/gen_golden.py | tee result.log
clean:
	rm -rf ./build
	rm -rf ./images/
	rm *.log