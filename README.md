# E2E_WordArt_DetRec
End2End pipeline for WordArt Text Detection and Recognition (Eng-Vie)

The pipeline is implemented for UIT-AIChallenge2022 consisting of text detection and text recognition. 
* For detection we use the Mask R-CNN (ResNext101-DCNv2_pretrainedImageNet) for precise segmentation. 
* For Recognition we use the VGG_Transformer for accurate OCR. 
<p>
  <img src="visualization/out_im0006.png" alt="Sample Image" style="height: 100%; width: 100%;">
</p>
* KHUYẾN-NGUYỄN
<p>
    <img src="visualization/out_im0011.png" alt="Sample Image" style="height: 50%; width: 50%;"><img src="visualization/out_im0014.png" alt="Sample Image" style="height: 50%; width: 50%;">
</p>

* CONVENTION-OMEGA-OMAGA

* SaleOff-XẢ-XẢ-THU-173-CHƯA-XẢ-100K-HÀNG-XẢ-BUỒN-ĐỒNG-10-120K-CỰC-SỐC-GIÁ-....
<p>
    <img src="visualization/out_im0012.png" alt="Sample Image" style="height: 50%; width: 50%;"><img src="visualization/out_im0013.png" alt="Sample Image" style="height: 50%; width: 50%;">
</p>

* TIỆM-TRÀ

* SALE-SALE-Collectpoint
# Environment Setup
```python 
conda create -n E2EWordArtDetRec python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate E2EWordArtDetRec
!pip install openmim
!mim install mmdet==2.25.0
!mim install mmocr
!mim install mmcv==1.6.0
%cd mmocr
!pip install -v -e .
%cd ../
```
# Data pre-processing

## Text Detection
For the detection task, we provide the data conversion from json to ICDAR2015 .txt style

The json dictionary format for each image polygon (multiple vertices):
```python
[
# Vietnamese encoder 
  {
    "text": "\u00d4",
    "points": [
      [52, 59],
      [228, 59],
      [228,262],
      [52,262]],
    "isArt": false
  },
# Unlabeled text   
  {
    "text": "###",
    "points": [
      [52, 59],
      [228, 59],
      [234, 62],
      [256, 65],
      [278, 66],
      [228,64],
      [52,262]],
    "isArt": false
  },
]
```

![Alt text](visualization/polygon_interpolation.png?raw=true "Polygon")

We interpolate multiple vertices polygon into 4 vertices polygon for easy segmentation

The .txt text ICDAR2015 style format for each image polygon (4 vertices):
```python
228,58,51,58,51,262,227,262,Ô
891,47,279,27,269,299,881,319,KÌA!
```
In order to change the annotation to ICDAR2015 (including interpolating the muti-vertice polygon and cleaning for unlabeled text), proceed the following code
```python
python tools/data_converter.py --label_root <root_anno_labels> --label_des <converted_output_anno_labels>
```
Ensuring the right data tree format

    Home
    ├── Dataset
    │   ├── images
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── labels
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing

After having converted to the ICDAR2015, we will use the configuration of MMOCR+MMDetection to train the text detector using the COCO format. We also provide the conversion to COCO format:
```python
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list training
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list validation
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list testing
```
Desired outcome

    Home
    ├── Dataset
    │   ├── images
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── labels
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── json_converted
    │   │   ├── instances_training.json
    │   │   ├── instances_validation.json
    │   │   ├── instances_testing.json

## Text Recognition
Crop the bounding boxes out as your wish. Modifying your datatree
    
    vietocr
    ├── data
    │   ├── images
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── ...
    │   │   ├── 3.png
    │   │   ├── annotation.txt
    
The annotation.txt folows the format of (No 2 words per image):
```python
data/images/1.png Ford
data/images/2.png PARA
data/images/3.png JUNIPER
data/images/4.png TABLE
...
```
# Training   
## Text Detection
```python
cd mmocr
python tools/train.py ./configuration/maskrcnn_resnext101_DCN_160e_icdar
```
## Text Recognition
We must prepare for the pretrained Vietnamese weights [weights](https://drive.google.com/file/d/13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA/view?usp=sharing):

    vietocr
    ├── weights
    │   ├── transformerocr.pth

Run the following code, modifying hyper-params in train.py:
```python
cd vietocr
python train.py
```
After running following comand, it will automatically convert data format to .lmdb and train your model

# Testing
