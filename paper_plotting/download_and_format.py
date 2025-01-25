import os
import shutil
import glob
from pathlib import Path
from huggingface_hub import snapshot_download

# # SAE Bench USE ONLY

# # This cell will download the SAE Bench results for graphing.

hf_repo_id = "adamkarvonen/sae_bench_results_0125"
local_dir = "./graphing_eval_results_0125"

if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        force_download=True,
        ignore_patterns=[
            "*autointerp_with_generations*",
            "*core_with_feature_statistics*",
        ],  # These use significant disk space / download time and are not needed for graphing
    )

    # SAE Bench USE ONLY

    # The purpose of this is that we currently organize results like this:

    # {eval_type}/{sae_release}/{sae_release}_{sae_id}_eval_results.json
    # because we have results for over 600 SAEs

    # However, the current scripts output results like this:
    # {eval_type}/{sae_release}_{sae_id}_eval_results.json

    # So, we just flatten the sae bench results to match the expected format

    # Get all immediate subdirectories in eval_results
    main_dirs = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]

    for main_dir in main_dirs:
        main_dir_path = os.path.join(local_dir, main_dir)
        print(f"\nProcessing {main_dir}...")

        # Get all subdirectories in the current directory
        subdirs = [
            d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))
        ]

        for subdir in subdirs:
            if not subdir.startswith("."):  # Skip hidden directories
                subdir_path = os.path.join(main_dir_path, subdir)
                print(f"Moving files from {subdir}")

                # Get all files in the subdirectory
                files = glob.glob(os.path.join(subdir_path, "*"))

                for file_path in files:
                    if os.path.isfile(file_path):  # Make sure it's a file, not a directory
                        file_name = os.path.basename(file_path)
                        destination = os.path.join(main_dir_path, file_name)

                        # Handle file name conflicts
                        if os.path.exists(destination):
                            base, extension = os.path.splitext(file_name)
                            counter = 1
                            while os.path.exists(destination):
                                new_name = f"{base}_{counter}{extension}"
                                destination = os.path.join(main_dir_path, new_name)
                                counter += 1

                        # Move the file
                        try:
                            shutil.move(file_path, destination)
                            print(
                                f"  Moved: {file_name} -> {os.path.basename(os.path.dirname(destination))}/"
                            )
                        except Exception as e:
                            print(f"  Error moving {file_name}: {str(e)}")

    print("\nFile moving complete!")
