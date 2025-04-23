#!/usr/bin/env python3
"""
Main script for simplified wildfire prediction project.
"""
import os
import argparse

def run_command(command):
    """Run a shell command."""
    print(f"Running: {command}")
    result = os.system(command)
    if result != 0:
        print(f"Warning: Command may have encountered an issue: {command}")
    return result == 0

def main():
    parser = argparse.ArgumentParser(description='Run wildfire prediction pipeline')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    parser.add_argument('--lookback', type=int, default=30, help='Days of weather history to use')
    parser.add_argument('--wildfire_file', type=str, default='../data/raw/ca_daily_fire_2000_2021-v2.csv', 
                       help='Path to consolidated California wildfires file')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization step')
    args = parser.parse_args()
    
    year = args.year
    lookback = args.lookback
    wildfire_file = args.wildfire_file
    
    print(f"\nRunning wildfire prediction pipeline for {year}")
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Check if wildfire file exists
    if not os.path.exists(wildfire_file):
        print(f"Error: Wildfire file not found at {wildfire_file}")
        print("Please make sure the file exists before running the pipeline.")
        return
    
    # 1. Data Loading
    print("\nStep 1: Data Loading")
    if not run_command(f"python data_loading.py --year={year} --wildfire_file={wildfire_file}"):
        print("Data loading failed. Please check the files and try again.")
        return
    
    # 2. Feature Engineering
    print("\nStep 2: Feature Engineering")
    if not run_command(f"python feature_engineering.py --year={year} --lookback={lookback}"):
        print("Feature engineering failed. Please check the previous step's output.")
        return
    
    # 3. Modeling
    print("\nStep 3: XGBoost Modeling")
    if not run_command(f"python modeling.py --year={year}"):
        print("Modeling failed. Please check the previous step's output.")
        return
    
    # 4. Visualization (new step)
    if not args.skip_visualization:
        print("\nStep 4: Creating Visualizations")
        if not run_command(f"python visualization.py --year={year}"):
            print("Visualization failed. Continuing with case studies...")
    
    # 5. Case Study Analysis
    print("\nStep 5: Case Study Analysis")
    if not run_command(f"python case_study.py --year={year} --count=5"):
        print("Case study analysis failed. Please check the previous steps' output.")
    
    print(f"\nPipeline completed for {year}")
    print(f"Results saved in ../results/{year}/ directory")

if __name__ == "__main__":
    main()