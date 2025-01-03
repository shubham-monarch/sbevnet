#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}Starting SBEVNet Simple Tests${NC}\n"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest is not installed. Installing pytest...${NC}"
    pip install pytest
fi

# Check if torch is installed
if ! python -c "import torch" &> /dev/null; then
    echo -e "${RED}PyTorch is not installed. Installing torch...${NC}"
    pip install torch
fi

# Check if numpy is installed
if ! python -c "import numpy" &> /dev/null; then
    echo -e "${RED}NumPy is not installed. Installing numpy...${NC}"
    pip install numpy
fi

echo -e "${GREEN}Running simple tests...${NC}\n"

# Run pytest with verbose output and show locals on failures
# Changed path to run from root directory
pytest sbevnet/models/test_bev_costvol_utils_simple.py -v --showlocals

# Capture the exit code
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}All simple tests passed successfully!${NC}"
else
    echo -e "\n${RED}Some simple tests failed. Please check the output above.${NC}"
fi

exit $TEST_EXIT_CODE 