#!/bin/bash

# ==============================================================================
# SemEval 2026 - Project Environment Setup Script
# Tasks: A, B, C
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- Styles ---
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

ENV_NAME="semeval"
CONFIG_FILE="environment.yml"

echo -e "${BOLD}${CYAN}🚀 Starting Project Setup for SemEval 2026...${NC}\n"

# -----------------------------------------------------------------------------
# 1. Directory Structure Initialization
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Initializing Project Structure...${NC}"

# Define Tasks
TASKS=("A" "B" "C")

# 1. Base folders
mkdir -p data img results src info_dataset

# 2. Loop through tasks to create specific structure
for task in "${TASKS[@]}"; do
    # Convention: data/Task_A, but results/results_TaskA, img/img_TaskA
    
    # Data Folder
    DATA_DIR="data/Task_${task}"
    if [ ! -d "$DATA_DIR" ]; then
        mkdir -p "$DATA_DIR"
        echo -e "  ├── Created Data: $DATA_DIR"
    fi

    # Image Folder
    IMG_DIR="img/img_Task${task}"
    if [ ! -d "$IMG_DIR" ]; then
        mkdir -p "$IMG_DIR"
        echo -e "  ├── Created Img:  $IMG_DIR"
    fi

    # Results Structure (Checkpoints, Inference, Submission)
    RESULTS_BASE="results/results_Task${task}"
    mkdir -p "$RESULTS_BASE/checkpoints"
    mkdir -p "$RESULTS_BASE/inference_output"
    mkdir -p "$RESULTS_BASE/submission"
    echo -e "  ├── Created Results structure for Task${task}"
done

echo -e "${GREEN}✔ Directory structure ready.${NC}\n"

# -----------------------------------------------------------------------------
# 2. Kaggle API Configuration
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Checking Kaggle Configuration...${NC}"

if [ -f "kaggle.json" ]; then
    echo -e "  Found 'kaggle.json' in root. Configuring..."
    
    # Create .kaggle directory if it doesn't exist
    mkdir -p ~/.kaggle
    
    # Copy file
    cp kaggle.json ~/.kaggle/kaggle.json
    
    # Set permissions (required by Kaggle API)
    chmod 600 ~/.kaggle/kaggle.json
    
    echo -e "${GREEN}✔ Kaggle API configured successfully.${NC}"
else
    echo -e "  [INFO] 'kaggle.json' not found in root. Skipping."
fi
echo ""

# -----------------------------------------------------------------------------
# 3. Environment Configuration (.env)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Checking .env Configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "   .env file not found. Creating a template..."
    
    cat <<EOT >> .env
# --- PROJECT PATHS ---
DATA_PATH=./data
IMG_PATH=./img

# --- COMET ML (Optional) ---
COMET_API_KEY=insert_your_key_here
COMET_PROJECT_NAME=semeval-2026
COMET_WORKSPACE=your_workspace_name

# --- MODEL CONFIG ---
# TOKENIZERS_PARALLELISM=false
EOT
    echo -e "${GREEN}✔ Created .env template. Please edit it.${NC}"
else
    echo -e "${GREEN}✔ .env file found.${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# 4. Package Manager Check
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Checking Package Manager...${NC}"

# Detect Conda or Mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo -e "  Mamba detected (Faster setup)."
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "  Conda detected."
else
    echo -e "${RED}Error: Neither Conda nor Mamba found.${NC}"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi
echo ""

# -----------------------------------------------------------------------------
# 5. Environment Installation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Setting up Python Environment '$ENV_NAME'...${NC}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# Check if environment already exists
if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo -e "  Environment '$ENV_NAME' exists. Updating from $CONFIG_FILE..."
    $CONDA_CMD env update -n $ENV_NAME -f $CONFIG_FILE --prune
else
    echo -e "  Creating new environment '$ENV_NAME' from $CONFIG_FILE..."
    $CONDA_CMD env create -f $CONFIG_FILE
fi

echo -e "\n${BOLD}${GREEN}Setup Completed Successfully!${NC}"
echo -e "${CYAN}-----------------------------------------------------${NC}"
echo -e "To start working, run:"
echo -e "${BOLD}conda activate $ENV_NAME${NC}"
echo -e "Then run your scripts (e.g., python src/src_TaskB/train.py)"
echo -e "${CYAN}-----------------------------------------------------${NC}"