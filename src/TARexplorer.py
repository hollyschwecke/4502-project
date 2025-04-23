import os
import tarfile
import glob

# Set the directory where tar files are located
tar_directory = '../data/raw'

# Find all .tar files in the directory
tar_pattern = os.path.join(tar_directory, "*.tar")
tar_files = glob.glob(tar_pattern)

if not tar_files:
    print(f"WARNING: No .tar files found with pattern: {tar_pattern}")
    # Try looking for other possible formats
    for ext in [".tar.gz", ".tgz", ".tar.bz2", ".tbz2"]:
        alt_pattern = os.path.join(tar_directory, f"*{ext}")
        alt_files = glob.glob(alt_pattern)
        if alt_files:
            print(f"Found {len(alt_files)} files with extension {ext}")
            tar_files = alt_files
            break

print(f"Found {len(tar_files)} archive files to extract")
for tar_file in tar_files:
    print(f"Processing: {tar_file}")

# Process each tar file
for tar_filename in tar_files:
    try:
        # Create directory name by removing extension(s)
        base_name = os.path.basename(tar_filename)
        if base_name.endswith('.tar'):
            dir_name = os.path.join(tar_directory, base_name[:-4])
        elif base_name.endswith(('.tar.gz', '.tgz')):
            dir_name = os.path.join(tar_directory, base_name.rsplit('.', 2)[0])
        elif base_name.endswith(('.tar.bz2', '.tbz2')):
            dir_name = os.path.join(tar_directory, base_name.rsplit('.', 2)[0])
        else:
            dir_name = os.path.join(tar_directory, os.path.splitext(base_name)[0])
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)
        
        # Extract the tar file to the directory
        print(f"Opening {tar_filename}...")
        with tarfile.open(tar_filename) as tar:
            # Get list of members to check if there's content
            members = tar.getmembers()
            print(f"  Found {len(members)} files in archive")
            
            if len(members) > 0:
                print(f"  Extracting to {dir_name}...")
                tar.extractall(path=dir_name)
                print(f"  Extraction complete")
            else:
                print(f"  WARNING: Archive appears to be empty")
        
        # Verify extraction
        extracted_files = os.listdir(dir_name)
        print(f"  Extracted {len(extracted_files)} items to {dir_name}/")
        if not extracted_files:
            print(f"  WARNING: Directory is empty after extraction!")
    
    except Exception as e:
        print(f"ERROR extracting {tar_filename}: {str(e)}")

print(f"Completed processing {len(tar_files)} archive files")