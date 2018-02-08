"# VisionCamExecutableWin" 

[README]

>> Variables at %PATH%:
C:\Users\tijuk\Envs\emotion\Scripts
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
C:\Program Files\Microsoft VS Code\bin
C:\Windows\system32
C:\Windows
C:\Windows\System32\Wbem
C:\Windows\System32\WindowsPowerShell\v1.0\
C:\Program Files\MeuScript\bin
C:\Program Files (x86)\GnuWin32\bin
C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common

//Relevant vars
C:\Users\tijuk\AppData\Local\Programs\Python\Python36
C:\Users\tijuk\AppData\Local\Programs\Python\Python36\Scripts
C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\
C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64
C:\Dev\MinGW\bin
C:\Users\tijuk\Envs\emotion\Lib\site-packages\scipy\extra-dll
C:\Users\tijuk\AppData\Local\Microsoft\WindowsApps

>> %PYTHONPATH%
C:\Users\tijuk\AppData\Local\Programs\Python\Python36

>> Versions:
Python: 3.6
MinGW : Thread model: win32. gcc version 6.3.0 (MinGW.org GCC-6.3.0-1)
opencv: 3.5

>> pip freeze
absl-py==0.1.10
altgraph==0.15
bleach==1.5.0
certifi==2018.1.18
chardet==3.0.4
Cython==0.26
future==0.16.0
futures==3.1.1
h5py==2.7.1
html5lib==0.9999999
httplib2==0.10.3
idna==2.6
imutils==0.4.5
Keras==2.1.3
macholib==1.9
Markdown==2.6.11
numpy==1.14.0
opencv-python==3.4.0.12
pefile==2017.11.5
protobuf==3.5.1
PyInstaller==3.3.1
pypiwin32==220
PyYAML==3.12
requests==2.18.4
scipy==1.0.0
six==1.11.0
tensorflow==1.5.0
tensorflow-tensorboard==1.5.0
urllib3==1.22
Werkzeug==0.14.1

>> Changes to scripts:
added:
[ from scipy import optimize ] to [ emotions.py ]

>> Changes to .spec from it's generic value (obtained by running pyinstaller emotions.py,
just make sure to delete both [dist] and [build] directories)

datas=[
    ('checkpoints\\epoch_75.hdf5','.\\checkpoints'),
    ('haarcascade_frontalface_default.xml','.'),
    ('emotions.mp4','.')
    ],
hiddenimports=['scipy._lib.messagestream'],
