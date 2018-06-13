# Computer-Vision-Utils
Repository for storing the following computer vision projects :

## 1. Object removal with seam carving
Remove selected object from input image by using [seam carving](https://en.wikipedia.org/wiki/Seam_carving) on both horizontal/vertical axes. 

![alt image](https://github.com/CretuCalin/Computer-Vision-Utils/blob/master/images/seam.png "Removing dog from picture")
#### Usage
```
# Create virtual environment
python3 -m venv venv

# Activate the environment 
source venv/bin/activate

# Install requirements 
pip install -r requirements.txt

# Remove object from picture
python seam.py --ref <image-path>
```

This script can also be used for content-aware image resizing.



## 2. Mosaic-Creator
Can be used to create [Photographic mosaic](https://en.wikipedia.org/wiki/Photographic_mosaic) based on euclidean distance between two pictures. 

![alt image](https://github.com/CretuCalin/Computer-Vision-Utils/blob/master/images/mosaic.png "Simpson image made of birds")

#### Usage
```
# Create virtual environment
python3 -m venv venv

# Activate the environment 
source venv/bin/activate

# Install requirements 
pip install -r requirements.txt

# Download and extract cifra images
./download_cifar.sh

# Create mosaic from picture
python mosaic.py --ref <image-path>
```

### Requirements
```numpy==1.14.4
opencv-python==3.4.1.15
mxnet==1.2.0
matplotlib==2.2.2
```
