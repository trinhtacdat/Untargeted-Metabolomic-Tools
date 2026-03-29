
# Read MGF Viewer code
with open('/mnt/data/MGF_Viewer_GUI.py', 'r') as f:
    mgf_code = f.read()

# Extract key MGF parsing functions
import re

# Find the MGF reading class and methods
mgf_class_match = re.search(r'class MGFViewer.*?(?=\nclass|\Z)', mgf_code, re.DOTALL)
if mgf_class_match:
    print("Found MGFViewer class")
    print("Length:", len(mgf_class_match.group()))
    
# Find specific methods
parse_mgf = re.search(r'def parse_mgf.*?(?=\n    def|\Z)', mgf_code, re.DOTALL)
if parse_mgf:
    print("\nFound parse_mgf method")
    print("Length:", len(parse_mgf.group()))

search_spectrum = re.search(r'def search_spectrum.*?(?=\n    def|\Z)', mgf_code, re.DOTALL)
if search_spectrum:
    print("\nFound search_spectrum method")
    print("Length:", len(search_spectrum.group()))

print("\n" + "="*60)
print("KEY MGF PARSING FUNCTIONS IDENTIFIED")
print("="*60)
