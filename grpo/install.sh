conda install -c conda-forge cudatoolkit-dev
pip install -e .
pip install vllm==0.7.3
pip install ray[default]
pip install transformers==4.49.0
pip install tensorboard omegaconf tensordict accelerate==0.34.2 codetiming==1.4.0 mathruler==0.1.0 pylatexenc==2.10 torchdata==0.11.0 datasets==2.16.1 wandb==0.18.6
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --upgrade
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./verl/utils/vllm_replace/* $site_pkg_path/vllm/
