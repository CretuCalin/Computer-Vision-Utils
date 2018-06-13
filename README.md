# Computer-Vision-Utils
Repository for storing the following computer vision projects :

## 1. Mosaic-Creator
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
