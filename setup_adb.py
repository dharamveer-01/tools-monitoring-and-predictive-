import os
import urllib.request
import zipfile
import subprocess

def install_adb():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    platform_tools_dir = os.path.join(base_dir, "platform-tools")
    adb_path = os.path.join(platform_tools_dir, "adb.exe")

    if os.path.exists(adb_path):
        print(f"✅ ADB is already installed at: {adb_path}")
        return True

    print("Downloading Android Platform Tools (ADB)...")
    url = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
    zip_path = os.path.join(base_dir, "platform-tools.zip")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
        
    os.remove(zip_path)
    print(f"✅ ADB successfully installed at: {adb_path}")
    return True

if __name__ == "__main__":
    install_adb()
