import zipfile
import os

def extract_dataset(zip_path, extract_to):
    """
    Extracts the dataset from a zip file.
    """
    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at {zip_path}")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: Bad zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define paths relative to the project root (assuming script runs from src or root context)
    # Adjusting to run from project root ideally
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_file = os.path.join(project_root, "diabetes_012_health_indicators_BRFSS2015.csv.zip")
    data_dir = os.path.join(project_root, "data")
    
    extract_dataset(zip_file, data_dir)
