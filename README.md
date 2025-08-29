https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal/summary?spm=a2c6h.13066369.question.10.786f3fc0ejBPJz


export work_dir=/mnt/sfs_turbo/
export container_work_dir=/mnt/sfs_turbo/
export container_name=botcall_aicall
export IMAGE=botcall:1.0
docker run -itd --privileged=true \
    --device=/dev/davinci0:/dev/davinci0 \
    --device=/dev/davinci1:/dev/davinci1 \
    --device=/dev/davinci2:/dev/davinci2 \
    --device=/dev/davinci3:/dev/davinci3 \
    --device=/dev/davinci4:/dev/davinci4 \
    --device=/dev/davinci5:/dev/davinci5 \
    --device=/dev/davinci6:/dev/davinci6 \
    --device=/dev/davinci7:/dev/davinci7 \
    --device=/dev/davinci_manager:/dev/davinci_manager \
    --device=/dev/devmm_svm:/dev/devmm_svm \
    --device=/dev/hisi_hdc:/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
    -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/vnpu.cfg:/etc/vnpu.cfg \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    --net=host \
    --cpus 192 \
    --memory 1000g \
    --shm-size 200g \
    -v ${work_dir}:${container_work_dir} \
    --name ${container_name} \
    $IMAGE \
    /bin/bash


source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

