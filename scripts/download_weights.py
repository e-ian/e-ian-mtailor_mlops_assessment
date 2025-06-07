"""
Script to download model weights
"""
import os
import urllib.request
import sys


def download_file(url, path, min_size=1000000):
    """Download file if it doesn't exist"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > min_size:
        print(f"{path} already exists")
        return True
    
    print(f"Downloading {path}...")

    try:
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) > min_size:
            print(f"Downloaded {path} ({os.path.getsize(path)} bytes)")
            return True
        else:
            print(f"Downloaded file too small")
            os.path.remove(path)
            return False
    except Exception as e:
        print("Failed to download {path}: {e}")
        return False

def create_placeholders():
    """Create placeholder files for testing."""
    # Model weights placeholder
    model_path = "models/pytorch_model_weights.pth"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(model_path):
        with open(model_path, 'w') as f:
            f.write("# Placeholder - download real weights with: python scripts/download_weights.py\n")
        print(f"Created placeholder: {model_path}")

    try:
        from PIL import Image
        os.makedirs("sample_images", exist_ok=True)
        
        images = [
            ("n01440764_tench.JPEG", (255, 165, 0)),
            ("n01667114_mud_turtle.JPEG", (34, 139, 34))
        ]
        
        for filename, color in images:
            path = f"sample_images/{filename}"
            if not os.path.exists(path):
                img = Image.new('RGB', (224, 224), color)
                img.save(path, 'JPEG')
                print(f"Created placeholder: {path}")
    except ImportError:
        print("PIL not available, skipping image placeholders")

def main():
    """Download model weights and sample images."""
    if len(sys.argv) > 1 and sys.argv[1] == "--placeholders":
        create_placeholders()
        return
    
    files = [
        (
         "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0",
         "models/pytorch_model_weights.pth", 40000000)
    ]
    
    success = True
    for url, path, min_size in files:
        if not download_file(url, path, min_size):
            success = False
    
    if success:
        print("All files ready!")
    else:
        print("Some downloads failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
