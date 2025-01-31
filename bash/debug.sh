#! /usr/bin/env bash

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            case="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case=${case:-case_1}  # Default to case_1 if no case specified
python3 debug.py --case "$case"