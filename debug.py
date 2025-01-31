#! /usr/bin/env python3

import logging
from typing import Callable, Dict
from dataclasses import dataclass
import traceback
from pathlib import Path
import argparse

from helpers import get_logger

logger = get_logger("debug")

@dataclass
class DebugCase:
    """Class to represent a debug test case"""
    name: str
    description: str
    function: Callable
    enabled: bool = True

class DebugRunner:
    """Class to manage and run debug test cases"""
    
    def __init__(self):
        self.cases: Dict[str, DebugCase] = {}
        self.debug_dir = Path("debug")
        
    def register_case(self, case_id: str, name: str, description: str, func: Callable, enabled: bool = True) -> None:
        """Register a new debug case"""
        self.cases[case_id] = DebugCase(name, description, func, enabled)
        
    def run_case(self, case_id: str) -> None:
        """Run a specific debug case"""
        if case_id not in self.cases:
            logger.error(f"Case {case_id} not found")
            return
            
        case = self.cases[case_id]
        if not case.enabled:
            logger.info(f"Case {case_id} ({case.name}) is disabled, skipping...")
            return
            
        logger.info(f"Running case {case_id}: {case.name}")
        logger.info(f"Description: {case.description}")
        logger.info("=" * 50)
        
        try:
            case.function()
        except Exception as e:
            logger.error(f"Error in case {case_id}: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("=" * 50)
        
    def run_all_enabled(self) -> None:
        """Run all enabled debug cases"""
        for case_id in self.cases:
            self.run_case(case_id)
            
    def get_case_dir(self, case_id: str) -> Path:
        """Get directory for debug case outputs"""
        case_dir = self.debug_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

# Define debug cases in separate module
from debug_cases import (
    test_labels_in_seg_masks
)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run debug test cases')
    parser.add_argument('--case', type=str, help='Specific case ID to run')
    args = parser.parse_args()

    # Initialize debug runner
    runner = DebugRunner()
    
    # Register all test cases
    cases = [
        ("case_1", "Dairy Masks", "Test dairy mask generation", test_labels_in_seg_masks),
    ]
    
    for case_id, name, desc, func in cases:
        runner.register_case(case_id, name, desc, func)
    
    # Run specific case if provided, otherwise run all cases
    if args.case:
        runner.run_case(args.case)
    else:
        runner.run_all_enabled()

if __name__ == "__main__":
    main()