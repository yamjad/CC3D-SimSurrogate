import os
import shutil

def delete_folders(base_path, threshold):
    """
    Deletes folders in the specified base_path whose names are numbers greater than the threshold.
    
    Parameters:
        base_path (str): The directory where the folders are located.
        threshold (int): The numerical threshold for folder names.
    """
    try:
        # List all items in the base path
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            # Check if the item is a directory and the name is a number
            if os.path.isdir(item_path):
                try:
                    folder_number = int(item)
                    if folder_number > threshold:
                        # Delete the folder
                        shutil.rmtree(item_path)
                        print(f"Deleted folder: {item_path}")
                except ValueError:
                    # Skip if the folder name is not a number
                    pass
    except Exception as e:
        print(f"Error: {e}")

# Example usage
base_path = "/u/pecenci/MIGRATION/Sims"  # Replace with the directory containing the folders
threshold = 683154-101

def confirm_threshold(threshold):
    confirmation = input(f"The threshold is set to {threshold}. Type 'yes' to proceed.")
    if confirmation.lower() == 'yes':
        print("Proceeding with execution...")
        return()
    else:
        print("Operation canceled.")
        exit()

# Example usage

confirm_threshold(threshold)

print("Proceeding with execution...")

delete_folders(base_path, threshold)