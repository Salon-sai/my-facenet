# Face Net In Xiao Yu

## Compatibility
The code is tested using Tensorflow 1.2 under Ubuntu 16.10 with Python 3.5

## Journey
| Date | Update |
|------|--------|
|2017-12-1 | Completed train Triplet Loss code. It can be used to train different CNNs Model|
|2017-12-6 | Completed the pipeline system|
|2017-12-15| Test the Pre-trained model (Inception-ResNet-v1 Model) in Xiao Yu Pipeline data set|

## Codes

- pipeline Fold : Pipeline In Xiao Yu
- model Fold: The CNNs models used to extract the image features

## Training data

- Small data set: [LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
- IMDB data set (Cropped): [IMDB_crop](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
- Wiki data set (Cropped): [WIKI_crop](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)

Also we can use the shell script to download the trainin data set:
```sh
./download.sh
```

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) | 0.987        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) | 0.992        | MS-Celeb-1M      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Todos

- [x] Begin the project and coding the major code heavily inspired by [FaceNet](https://github.com/davidsandberg/facenet)
- [x] Try to train the MobileNet with triplet loss
- [x] Test the pre-trained model (Face net) in Pipeline
- [ ] Use Pre-trained model as data extraction and train gender with last full connection after data extraction, using Softmax function as training function


