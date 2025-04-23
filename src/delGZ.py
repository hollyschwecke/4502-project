import os
import glob
from pathlib import Path

def delete_gz_files(base_dir):
    """
    Recursively find and delete all .op.gz files in the base_dir and its subdirectories,
    but only if the corresponding .op file exists
    """
    # Count for statistics
    total_gz_files = 0
    deleted_files = 0
    skipped_files = 0
    
    # Find all directories that match the pattern (gsod_YYYY)
    gsod_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('gsod_')]
    
    print(f"Found {len(gsod_dirs)} gsod directories")
    
    # Process each directory
    for gsod_dir in gsod_dirs:
        dir_path = os.path.join(base_dir, gsod_dir)
        print(f"Processing directory: {dir_path}")
        
        # Find all .op.gz files in this directory
        gz_files = glob.glob(os.path.join(dir_path, "*.op.gz"))
        total_gz_files += len(gz_files)
        
        # Process each gzip file
        for gz_file in gz_files:
            try:
                # Check if the corresponding .op file exists
                op_file = os.path.splitext(gz_file)[0]  # Remove .gz extension
                
                if os.path.exists(op_file):
                    # Delete the .gz file
                    os.remove(gz_file)
                    deleted_files += 1
                    
                    # Print progress every 100 files
                    if deleted_files % 100 == 0:
                        print(f"Deleted {deleted_files}/{total_gz_files} files...")
                else:
                    # Skip deletion since the extracted file doesn't exist
                    print(f"Warning: Skipping {gz_file} because {op_file} doesn't exist")
                    skipped_files += 1
                
            except Exception as e:
                print(f"Error deleting {gz_file}: {str(e)}")
                skipped_files += 1
    
    print(f"Completed! Deleted {deleted_files} out of {total_gz_files} .op.gz files")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files (missing corresponding .op files or errors)")

if __name__ == "__main__":
    # Change this to the directory containing your gsod_YYYY folders
    base_directory = '../data/raw'
    
    # Safety confirmation
    print("This script will delete all .op.gz files that have corresponding .op files.")
    confirm = input("Are you sure you want to continue? (y/n): ")
    
    if confirm.lower() == 'y':
        delete_gz_files(base_directory)
    else:
        print("Operation cancelled.")