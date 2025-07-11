import os
import shutil

# Check if Tesseract is installed
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    print(f"Tesseract executable found at: {tesseract_path}")
else:
    print("Tesseract executable not found. Please make sure it is installed.")

# Check the default Tesseract path
default_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'/usr/bin/tesseract',
    r'/usr/local/bin/tesseract',
]

for path in default_paths:
    if os.path.exists(path):
        print(f"Default Tesseract path: {path}")
        break
else:
    print("Default Tesseract path not found.")
