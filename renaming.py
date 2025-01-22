import os
import re

# Set the base path to the folder containing 'graphing_eval_results_0119'
base_path = os.path.join(os.path.dirname(__file__), 'graphing_eval_results_0119')

# Function to rename files based on the pattern
def rename_files_recursively(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                # Match the trainer index pattern
                match = re.search(r'trainer_(\d+)', file)
                if match:
                    idx = int(match.group(1))
                    new_idx = idx % 6
                    # Create the new filename
                    new_file = re.sub(r'trainer_\d+', f'trainer_{new_idx}', file)
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_file)
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

# Execute the function
rename_files_recursively(base_path)