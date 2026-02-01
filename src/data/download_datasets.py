"""Dataset Download Script"""
import os
import urllib.request

def create_directories():
    os.makedirs('data/raw/uci_heart', exist_ok=True)
    print("✓ Directories created")

def download_uci_heart_disease():
    print("\nDownloading UCI Heart Disease Dataset...")
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    files = ['processed.cleveland.data']
    
    for file in files:
        url = base_url + file
        save_path = f'data/raw/uci_heart/{file}'
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"✓ Downloaded {file}")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    create_directories()
    download_uci_heart_disease()
    print("\n✓ Download complete!")
