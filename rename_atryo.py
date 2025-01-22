import os
import re

# Set the base path to the folder containing 'graphing_eval_results_0119'
base_path = os.path.join(os.path.dirname(__file__), 'matroyshka_eval_results_0117')

# Function to rename files based on the pattern
def rename_atroys_to_atryos(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'atroys' in file:  # Check if 'atroys' exists in the filename
                # Replace all occurrences of 'atroys' with 'atryos'
                new_file = file.replace('atroys', 'atryos')
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_file)
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# Execute the function
rename_atroys_to_atryos(base_path)
