Some examples if something goes wrong, the error message, and what I did to make it work.


> cv2.error: C:\projects\opencv-python\opencv\modules\imgproc\src\color.cpp:11111: error: (-215) scn == 3 || scn == 4 in function cv::cvtColor
Im my case, it was my Camera being connected to my VirtualMachine, I just had to unplug it and plug it back in.

> AttributeError: 'NoneType' object has no attribute 'shape'
