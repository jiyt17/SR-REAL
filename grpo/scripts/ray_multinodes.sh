#!/bin/bash
set -x  # Enable debugging
train_script=$1

DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=25001

echo "======= RAY MULTINODES SCRIPT STARTING ======="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "PWD = $(pwd)"
echo "Train script = $train_script"

NNODES=${SLURM_JOB_NUM_NODES:-1}
echo "NNODES = $NNODES"

# Check if scontrol is available, if not use fallback
if command -v scontrol &> /dev/null; then
    NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    echo "Warning: scontrol not found, using fallback node detection"
    echo "DEBUG: SLURM_JOB_NODELIST = '$SLURM_JOB_NODELIST'"
    if [[ $SLURM_JOB_NODELIST =~ ^([^[]+)\[([0-9]+) ]]; then
        PREFIX=${BASH_REMATCH[1]}
        FIRST_NUM=${BASH_REMATCH[2]}
        MASTER_ADDR="${PREFIX}${FIRST_NUM}"
        echo "Bracket notation detected: PREFIX='$PREFIX', FIRST_NUM='$FIRST_NUM'"
        echo "Result: MASTER_ADDR='$MASTER_ADDR'"
    elif [[ $SLURM_JOB_NODELIST =~ ^([^,\[]+) ]]; then
        MASTER_ADDR=${BASH_REMATCH[1]}
        echo "Simple node name detected: $MASTER_ADDR"
    else
        # Extract first node from simple comma-separated list
        MASTER_ADDR=$(echo $SLURM_JOB_NODELIST | sed 's/,.*//')
        echo "Fallback extraction: $MASTER_ADDR"
    fi
    NODES=$SLURM_JOB_NODELIST
    echo "Extracted MASTER_ADDR from SLURM_JOB_NODELIST: $MASTER_ADDR"
fi

echo "NODES = $NODES"

NODE_RANK=${SLURM_PROCID:-0}
echo "NODE_RANK = $NODE_RANK"

MASTER_ADDR=${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}
echo "MASTER_ADDR = $MASTER_ADDR"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"

# Setup conda environment first, before any Ray operations
echo "======= SETTING UP CONDA ENVIRONMENT ======="
echo "Activating conda environment..."
source path/to/anaconda3/etc/profile.d/conda.sh
conda activate verl

# Set HuggingFace cache directory to avoid permission issues
export HF_HOME="path/to/svila-verl/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"

echo "HuggingFace cache directory set to: $HF_HOME"

echo "Conda environment activated:"
echo "Python: $(which python) - $(python --version)"
echo "Ray: $(which ray) - $(ray --version)"

# Check if ray is available, install if needed#!/bin/bash
set -x  # Enable debugging
train_script=$1

DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=25001

echo "======= RAY MULTINODES SCRIPT STARTING ======="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "PWD = $(pwd)"
echo "Train script = $train_script"

NNODES=${SLURM_JOB_NUM_NODES:-1}
echo "NNODES = $NNODES"

# Check if scontrol is available, if not use fallback
if command -v scontrol &> /dev/null; then
    NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    echo "Warning: scontrol not found, using fallback node detection"
    echo "DEBUG: SLURM_JOB_NODELIST = '$SLURM_JOB_NODELIST'"
    if [[ $SLURM_JOB_NODELIST =~ ^([^[]+)\[([0-9]+) ]]; then
        PREFIX=${BASH_REMATCH[1]}
        FIRST_NUM=${BASH_REMATCH[2]}
        MASTER_ADDR="${PREFIX}${FIRST_NUM}"
        echo "Bracket notation detected: PREFIX='$PREFIX', FIRST_NUM='$FIRST_NUM'"
        echo "Result: MASTER_ADDR='$MASTER_ADDR'"
    elif [[ $SLURM_JOB_NODELIST =~ ^([^,\[]+) ]]; then
        MASTER_ADDR=${BASH_REMATCH[1]}
        echo "Simple node name detected: $MASTER_ADDR"
    else
        # Extract first node from simple comma-separated list
        MASTER_ADDR=$(echo $SLURM_JOB_NODELIST | sed 's/,.*//')
        echo "Fallback extraction: $MASTER_ADDR"
    fi
    NODES=$SLURM_JOB_NODELIST
    echo "Extracted MASTER_ADDR from SLURM_JOB_NODELIST: $MASTER_ADDR"
fi

echo "NODES = $NODES"

NODE_RANK=${SLURM_PROCID:-0}
echo "NODE_RANK = $NODE_RANK"

MASTER_ADDR=${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}
echo "MASTER_ADDR = $MASTER_ADDR"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"

# # Setup conda environment first, before any Ray operations
# echo "======= SETTING UP CONDA ENVIRONMENT ======="
# echo "Activating conda environment..."
# source path/to/anaconda3/etc/profile.d/conda.sh
# conda activate verl

# # Set HuggingFace cache directory to avoid permission issues
# # export HF_HOME="path/to/svila-verl/.cache/huggingface"
# export HF_HOME="path/to/svila-verl/.cache/huggingface"
# export TRANSFORMERS_CACHE="$HF_HOME"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
# export HF_MODULES_CACHE="$HF_HOME/modules"

# echo "HuggingFace cache directory set to: $HF_HOME"

# echo "Conda environment activated:"
# echo "Python: $(which python) - $(python --version)"
# echo "Ray: $(which ray) - $(ray --version)"

# Check if ray is available, install if needed
echo "======= CHECKING RAY INSTALLATION ======="
if ! command -v ray &> /dev/null; then
    echo "Ray not found, installing..."
    pip install ray[default]
    echo "Ray installation completed"
else
    echo "Ray is already available"
fi

echo "======= RAY VERSION CHECK ======="
ray --version

echo "======= CLEANING UP EXISTING RAY PROCESSES ======="
# Kill any existing Ray processes to avoid port conflicts
pkill -f "ray::" || true
pkill -f "raylet" || true
pkill -f "gcs_server" || true
sleep 3

# Use job ID to create unique ports to avoid conflicts between concurrent jobs
RAY_PORT_BASE=$((6000 + (SLURM_JOB_ID % 1000)))
RAY_GCS_PORT=$RAY_PORT_BASE
RAY_DASHBOARD_PORT=$((RAY_PORT_BASE + 100))
RAY_DASHBOARD_AGENT_PORT=$((RAY_PORT_BASE + 200))

echo "======= STARTING RAY CLUSTER SETUP ======="
echo "Using ports: GCS=$RAY_GCS_PORT, Dashboard=$RAY_DASHBOARD_PORT, Agent=$RAY_DASHBOARD_AGENT_PORT"

# Create a shared file to communicate the Ray head IP (use shared filesystem)
RAY_IP_FILE="path/to/svila-verl/ray_head_ip_${SLURM_JOB_ID}"

if [ $NODE_RANK -eq 0 ]; then
    echo "======= STARTING RAY HEAD NODE (rank 0) ======="
    echo "Starting Ray head on $MASTER_ADDR:$RAY_GCS_PORT"
    
    # Create IP file early to unblock workers, using fallback IP
    echo "Creating preliminary IP file to unblock workers..."
    RAY_HEAD_IP=$(python3 -c "import socket; print(socket.gethostbyname('$MASTER_ADDR'))" 2>/dev/null || echo "$MASTER_ADDR")
    echo "$RAY_HEAD_IP" > $RAY_IP_FILE
    echo "Preliminary IP file created: $RAY_IP_FILE with IP: $RAY_HEAD_IP"
    
    # Start Ray head with unique ports
    echo "Starting Ray head process..."
    ray start --head --port=$RAY_GCS_PORT \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --dashboard-agent-grpc-port=$RAY_DASHBOARD_AGENT_PORT > path/to/svila-verl/ray_head_output_${SLURM_JOB_ID}.log 2>&1 &
    RAY_HEAD_PID=$!
    echo "Ray head started with PID: $RAY_HEAD_PID"
    sleep 15
    
    # Extract the actual IP address from Ray's output and update if different
    ACTUAL_RAY_HEAD_IP=$(grep "Local node IP:" path/to/svila-verl/ray_head_output_${SLURM_JOB_ID}.log | awk '{print $NF}' | tail -1)
    if [ -n "$ACTUAL_RAY_HEAD_IP" ] && [ "$ACTUAL_RAY_HEAD_IP" != "$RAY_HEAD_IP" ]; then
        echo "Updating IP file with actual Ray IP: $ACTUAL_RAY_HEAD_IP"
        echo "$ACTUAL_RAY_HEAD_IP" > $RAY_IP_FILE
        RAY_HEAD_IP="$ACTUAL_RAY_HEAD_IP"
    fi
    
    echo "Final Ray head IP address: $RAY_HEAD_IP"
    echo "IP file content: $(cat $RAY_IP_FILE)"
    ls -la $RAY_IP_FILE
    
    echo "======= RAY CLUSTER STATUS ======="
    ray status || echo "Ray status check failed"
    
    echo "======= STARTING TRAINING SCRIPT ======="
    echo "Running: nnodes=$NNODES bash $train_script"
    echo "Checking if training script exists..."
    if [ -f "$train_script" ]; then
        echo "Training script found: $train_script"
        ls -la "$train_script"
    else
        echo "ERROR: Training script not found: $train_script"
        echo "Current directory: $(pwd)"
        echo "Directory contents:"
        ls -la
        exit 1
    fi
    
    echo "======= EXECUTING TRAINING SCRIPT ======="
    # Set Ray address to connect to existing cluster 
    export RAY_ADDRESS="${RAY_HEAD_IP}:$RAY_GCS_PORT"
    echo "Setting RAY_ADDRESS=$RAY_ADDRESS"
    
    echo "Ray environment variables:"
    env | grep RAY
    
    # Ensure conda environment is available for the training script
    echo "Final environment check before training:"
    echo "Python: $(which python) - $(python --version)"
    echo "Ray: $(which ray) - $(ray --version)"
    
    nnodes=$NNODES bash -x $train_script
    TRAIN_EXIT_CODE=$?
    echo "======= TRAINING COMPLETED WITH EXIT CODE: $TRAIN_EXIT_CODE ======="
    
    # Clean up
    rm -f $RAY_IP_FILE
else
    echo "======= STARTING RAY WORKER NODE (rank $NODE_RANK) ======="
    
    # Wait for the Ray head IP file to be created with timeout
    echo "Waiting for Ray head IP file at $RAY_IP_FILE..."
    WAIT_COUNT=0
    MAX_WAIT=300  # 5 minutes maximum wait
    while [ ! -f "$RAY_IP_FILE" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        echo "Still waiting for Ray head IP file... ($WAIT_COUNT/$MAX_WAIT)"
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    if [ ! -f "$RAY_IP_FILE" ]; then
        echo "ERROR: Ray head IP file never created after $MAX_WAIT attempts. Exiting worker."
        exit 1
    fi
    
    RAY_HEAD_IP=$(cat $RAY_IP_FILE)
    echo "Found Ray head IP: $RAY_HEAD_IP"
    
    # Wait until head node is ready using Python with timeout
    echo "Waiting for Ray head at ${RAY_HEAD_IP}:$RAY_GCS_PORT..."
    CONNECT_COUNT=0
    MAX_CONNECT=150  # 5 minutes maximum wait for connection
    while ! python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$RAY_HEAD_IP', $RAY_GCS_PORT)); s.close()" 2>/dev/null && [ $CONNECT_COUNT -lt $MAX_CONNECT ]; do
        echo "Still waiting for Ray head at ${RAY_HEAD_IP}:$RAY_GCS_PORT... ($CONNECT_COUNT/$MAX_CONNECT)"
        sleep 2
        CONNECT_COUNT=$((CONNECT_COUNT + 1))
    done
    
    if [ $CONNECT_COUNT -ge $MAX_CONNECT ]; then
        echo "ERROR: Could not connect to Ray head after $MAX_CONNECT attempts. Exiting worker."
        exit 1
    fi
    
    echo "Ray head is ready, connecting worker node..."
    ray start --address=${RAY_HEAD_IP}:$RAY_GCS_PORT
    echo "Worker node connected to Ray cluster at ${RAY_HEAD_IP}:$RAY_GCS_PORT"
    
    # Verify connection by checking ray status
    echo "Verifying Ray worker connection..."
    ray status || echo "Ray status check failed on worker"
    
    # Keep the worker running
    echo "Worker node is running, waiting for job completion..."
    while [ -f "$RAY_IP_FILE" ]; do
        sleep 10
    done
    echo "Job completed, worker node shutting down..."
fi

echo "======= RAY MULTINODES SCRIPT COMPLETED ======="
