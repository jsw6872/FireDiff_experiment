import subprocess

p1 = subprocess.Popen("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'", stdout=subprocess.PIPE, shell=True)
p2 = subprocess.Popen("pip install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113", stdout=subprocess.PIPE, shell=True)
p3 = subprocess.Popen("pip install -U numpy", stdout=subprocess.PIPE, shell=True)

print(p1.communicate())
print(p2.communicate())
print(p3.communicate())