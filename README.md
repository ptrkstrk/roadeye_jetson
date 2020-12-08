# roadeye_jetson
Real-time traffic sign detection on Jetson Nano platform. The module in this repository is responsible for performing the detection on a live camera image and sending the detection data to user's phone. This repository i a part of a traffic sign detection system developed for my bachelor thesis. roadeye_android directory contains the Android app code that notifies the user about signs. Running the camera_inference.py script results in starting the Bluetooth server and running detection. detection_model contains code and notebooks used for training the model. RetinaNet architecture was used. It is implemented in Detectron2 library (https://github.com/facebookresearch/detectron2). The model was trained on Mapillary traffic signs dataset (https://www.mapillary.com/dataset/trafficsign). As the dataset was provided to me only for research purposes, the model file is missing from this repository.

# Required libraries on Jetson
To run detection on Jetson these libraries need to be installed:
1. PyTorch with torchvision. Instructions are provided here: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
2. Detectron2. It needs to be compiled from source, as described here: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues
3. Numpy
4. PyBluez. For the system to work on Jetson, the  Bluetooth  daemon  needs  to  run  in  compatibility  mode. To do that, file /lib/systemd/system/bluetooth.service needs to be modified by adding ’-C’ after ’bluetoothd’.
5. OpenCV - Jetson Hacks provides excellent tutorial: https://www.jetsonhacks.com/2019/11/22/opencv-4-cuda-on-jetson-nano/

