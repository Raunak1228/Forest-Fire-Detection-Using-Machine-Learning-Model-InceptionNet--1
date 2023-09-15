import os

folder_path = "C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/Thermal Images Final Data/Test/No Fire"  # Specify the path to your folder here

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter out only the image files
image_files = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# Sort the image files in ascending order
image_files.sort()

# Rename the image files with the desired format
for i, file_name in enumerate(image_files):
    # Split the file name and extension
    name, ext = os.path.splitext(file_name)

    # Create the new name using the desired format (e.g., img_1, img_2, etc.)
    new_name = f"img_{i+1}{ext}"

    # Construct the full file paths for the original and new names
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    print(f"Renamed {file_name} to {new_name}")
