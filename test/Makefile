# ----------------------------------------------------------------
# Variables setup Section                                        -
# ----------------------------------------------------------------
SCRIPT =             ../src/clustering_eval.py
RUN_DIR =            ./run_dir
RUN_FILES =          ../data/run_files
RESULTS_DIR =        $(RUN_DIR)/results
INPUT_DATA_DIR =     ./data
UNIT_TEST_TEMP_DIR = ./unit/tmp
DATA_DIR =           ../data/spreadsheets
INPUT_DATA =         $(foreach dir,$(DATA_DIR),$(wildcard $(dir)/*))
INPUT_DATA_SUFFIX =  $(suffix $(INPUT_DATA))

RMR =                rm -rf
GZIP =               gzip
GZIP_D =             gzip -d
MKDIR_P =            mkdir -p

# ----------------------------------------------------------------
# Benchmarks setup Section                                       -
# ----------------------------------------------------------------
env_setup: create_run_dir copy_run_files 

create_run_dir:
	$(MKDIR_P) $(RESULTS_DIR) 

copy_run_files:
	cp $(RUN_FILES)/*.yml $(RUN_DIR) 

run_cluster_eval:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_1_cluster_eval.yml

# ----------------------------------------------------------------
# - CLEAN UP SECTION                                             -
# ----------------------------------------------------------------	
final_clean: clean_dir_recursively 

clean_dir_recursively:
	$(RMR) $(RUN_DIR)	
