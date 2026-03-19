# export HF_HOME=/cache
export HF_HOME=/cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME/transformers

source ~/venvs/eval_finance/bin/activate

pip list

PROJECT_PATH='.'
OPENAI_API_KEY='your_openai_api_key'
ANTHROPIC_API_KEY='your_anthropic_api_key'
GEMINI_API_KEY='your_gemini_api_key'
DEEPSEEK_API_TOKEN='your_deepseek_api_key'
export PROJECT_PATH=${PROJECT_PATH}
export OPENAI_API_KEY=${OPENAI_API_KEY}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export DEEPSEEK_API_TOKEN=${DEEPSEEK_API_TOKEN}

gpu_id=0
sys_msg="On"
dataset="XFinBench"
task="bool"
# model="micdun/fino1_qwen2_5-1_5b"
# model="micdun/med_r1_qwen2_5-2x1_5b"
# model="micdun/medr1_fino1_qwen2_5-3x1_5b"
# model="Qwen/Qwen2.5-1.5B-Instruct"
# model="micdun/unrestricted_fino1_qwen2_5-1_5b-ft" # normal finance one
# model="micdun/unrestricted_medr1_qwen2_5-1_5b-ft" # normal med one
# model="micdun/unrestricted_medr1_fino1_qwen2_5-3x1_5b" # unrestricted MoE
model="Qwen/Qwen2.5-7B-Instruct"
# model="alrope/merged_fino1_medr1_ind_qwen2_5-1_5b-ft"
reason_type="CoT"
retri_type="something"
retriever="bm25"
top_k_retr=3
max_token=16384

if [ -n "$SSL_CERT_FILE" ] && [ ! -f "$SSL_CERT_FILE" ]; then
  unset SSL_CERT_FILE
fi

CUDA_VISIBLE_DEVICES=$gpu_id python ${PROJECT_PATH}/evaluate/main.py \
    --dataset $dataset\
    --task $task\
    --model $model\
    --reason_type $reason_type\
    --sys_msg $sys_msg\
    --retri_type $retri_type\
    --retriever $retriever\
    --top_k_retr $top_k_retr\
    --max_token $max_token \
    --batch_size 16
    