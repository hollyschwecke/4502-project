import os
import gzip
import shutil
import glob
from pathlib import Path

def extract_gzip_files(base_dir):
    """
    Recursively find and extract all .op.gz files in the base_dir and its subdirectories
    """
    # Count for statistics
    total_files = 0
    extracted_files = 0
    
    # Find all directories that match the pattern (gsod_YYYY)
    gsod_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('gsod_')]
    
    print(f"Found {len(gsod_dirs)} gsod directories")
    
    # Process each directory
    for gsod_dir in gsod_dirs:
        dir_path = os.path.join(base_dir, gsod_dir)
        print(f"Processing directory: {dir_path}")
        
        # Find all .op.gz files in this directory
        gz_files = glob.glob(os.path.join(dir_path, "*.op.gz"))
        total_files += len(gz_files)
        
        # Process each gzip file
        for gz_file in gz_files:
            try:
                # Output file name (remove .gz extension)
                output_file = os.path.splitext(gz_file)[0]
                
                # Extract the gzip file
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                extracted_files += 1
                
                # Optional: remove the original .gz file after extraction
                # os.remove(gz_file)
                
                # Print progress every 100 files
                if extracted_files % 100 == 0:
                    print(f"Extracted {extracted_files}/{total_files} files...")
                
            except Exception as e:
                print(f"Error extracting {gz_file}: {str(e)}")
    
    print(f"Completed! Extracted {extracted_files} out of {total_files} files")

if __name__ == "__main__":
    # Change this to the directory containing your gsod_YYYY folders
    base_directory = '../data/raw'
    extract_gzip_files(base_directory)