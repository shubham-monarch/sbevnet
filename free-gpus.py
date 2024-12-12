#!/usr/bin/env python3

import os
import subprocess
import sys

from helpers import get_logger

logger = get_logger('free-gpus')


def kill_gpu_processes() -> None:
    """Kill all Python processes using GPUs."""
    try:
        # Get all processes using nvidia-smi
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        ).decode()
        
        # Get list of PIDs
        gpu_pids = [int(pid) for pid in nvidia_smi.strip().split('\n') if pid]
        
        if not gpu_pids:
            logger.info(f"===================")
            logger.info("No GPU processes found.")
            logger.info(f"===================\n")
            return
        
        logger.error(f"===================")
        logger.error(f"Found {len(gpu_pids)} GPU processes to kill...")
        logger.error(f"===================\n")

        # Kill each process
        for pid in gpu_pids:
            try:
                os.kill(pid, 9)  # SIGKILL
                logger.error(f"Killed process {pid}")
            except ProcessLookupError:
                logger.error(f"Process {pid} already terminated")
            except PermissionError:
                logger.error(f"Permission denied to kill process {pid}. Try running with sudo.")
        
        logger.error(f"\nAll GPU processes have been terminated.")
        
    except FileNotFoundError:
        logger.error("Error: nvidia-smi not found. Is NVIDIA driver installed?")
    except subprocess.CalledProcessError:
        logger.error("Error running nvidia-smi command")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    kill_gpu_processes() 