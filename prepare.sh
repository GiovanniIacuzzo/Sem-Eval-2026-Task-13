#!/bin/bash

# ==============================================================================
# SemEval 2026 Task 13 - Environment Setup Script
# Optimized for macOS Apple Silicon
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

echo -e "${BOLD}${CYAN}🚀 Starting Project Setup for SemEval 2026 Task 13...${NC}\n"

# -----------------------------------------------------------------------------
# 1. Directory Structure Initialization
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/4] Initializing Project Structure...${NC}"

DIRS=(
    "results/logs"
    "results/checkpoints"
    "results/submission"
    "results/inference_analysis"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  ├── Created: $dir"
    else
        echo -e "  ├── Exists:  $dir"
    fi
done
echo -e "${GREEN}✔ Structure ready.${NC}\n"

# -----------------------------------------------------------------------------
# 2. Environment Configuration (.env)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/4] Checking Configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "   .env file not found. Creating a template..."
    
    cat <<EOT >> .env
# --- PROJECT PATHS ---
DATA_PATH=./data
# TEST_DATA_PATH=./data/test.parquet # Uncomment if custom path needed

# --- COMET ML ---
COMET_API_KEY=insert_your_key_here
COMET_PROJECT_NAME=semeval-task13
COMET_WORKSPACE=your_workspace_name
COMET_EXPERIMENT_NAME=code_classification_v1

# --- TOKENIZERS ---
TOKENIZERS_PARALLELISM=false
EOT
    echo -e "${GREEN}✔ Created .env template. Please edit it with your API keys.${NC}"
else
    echo -e "${GREEN}✔ .env file found.${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# 3. Conda/Mamba Checks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/4] Checking Package Manager...${NC}"

# Detect Conda or Mamba (Mamba is faster if installed)
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo -e "  Mamba detected (Faster setup)."
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "  Conda detected."
else
    echo -e "${RED}Error: Neither Conda nor Mamba found.${NC}"
    echo "Please install Miniconda or Miniforge (recommended for M2) first."
    exit 1
fi
echo ""

# -----------------------------------------------------------------------------
# 4. Environment Installation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/4] Setting up Python Environment '$ENV_NAME'...${NC}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# Check if environment already exists
if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo -e "  Environment '$ENV_NAME' exists. Updating..."
    $CONDA_CMD env update -n $ENV_NAME -f $CONFIG_FILE --prune
else
    echo -e "  Creating new environment '$ENV_NAME'..."
    $CONDA_CMD env create -f $CONFIG_FILE
fi

echo -e "\n${BOLD}${GREEN}Setup Completed Successfully!${NC}"
echo -e "${CYAN}-----------------------------------------------------${NC}"
echo -e "To start working, run:"
echo -e "${BOLD}conda activate $ENV_NAME${NC}"
echo -e "${CYAN}-----------------------------------------------------${NC}"