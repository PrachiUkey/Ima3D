import os

# Path to the grandparent folder
grandparent_folder = "model"

# Iterate over each parent folder (e.g., bed, bookcase)
for parent_folder in os.listdir(grandparent_folder):
    parent_path = os.path.join(grandparent_folder, parent_folder)
    if not os.path.isdir(parent_path):
        continue  # skip files, only process folders

    # List all subfolders inside this parent folder
    subfolders = [f for f in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, f))]
    subfolders.sort()  # optional: sort alphabetically

    # Rename each subfolder
    for idx, folder_name in enumerate(subfolders):
        old_path = os.path.join(parent_path, folder_name)
        new_name = f"{parent_folder}_{idx}"
        new_path = os.path.join(parent_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{old_path}' -> '{new_path}'")

print("All subfolders renamed successfully.")
