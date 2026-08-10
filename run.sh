export http_proxy="http://127.0.0.1:15014"
export https_proxy="http://127.0.0.1:15014"
export ftp_proxy="http://127.0.0.1:15014"
export no_proxy="localhost,http://127.0.0.1"

conda activate tlx
source set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ./build
make -j 128
if [ $? -ne 0 ]; then
    exit 1
fi
cd ../
export PYTHONPATH=$PYTHONPATH:/home/dyq/tilelang-ascend/tilelang/analysis/

# python ./examples/flash_attention/flash_attn_bhsd_cc_sync.py
python ./examples/gemm/example_gemm.py
# python ./examples/sparse_flash_attention/example_sparse_flash_attn.py

# rm -rf build/
# mkdir build && cd build
# cmake ..