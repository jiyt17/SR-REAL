cd path/to/svila-verl/eval
conda activate sr-3d


torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_sparbench.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_spar \
  --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_sparbench_region.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_spar_region \
  --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_embspatial.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_embspatial \
  --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_embspatial_region.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_embspatial_region \
  --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_sat.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_sat \
  --reason 

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_blink.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_blink 
  # --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_cvbench.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_cvbench 
  # --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_rwqa.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_rwqa
  # --reason

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12355 \
  eval_erqa.py \
  --model-path path/to/SR-real-model \
  --output-dir ./results/SR-real-model_erqa
  # --reason