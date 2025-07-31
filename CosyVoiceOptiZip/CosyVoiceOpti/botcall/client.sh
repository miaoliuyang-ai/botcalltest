source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH
python3 client.py