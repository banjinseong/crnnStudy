import os
import lmdb
import cv2
import numpy as np
import six
import sys
from PIL import Image
from tqdm import tqdm

def check_image_is_valid(imageBin):
    if imageBin is None:
        return False
    imageBuf = six.BytesIO()
    imageBuf.write(imageBin)
    imageBuf.seek(0)
    try:
        img = Image.open(imageBuf).convert('L')
        img.verify()
    except:
        return False
    return True

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(imageFolderPath, labelFolderPath, outputPath):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        imageFolderPath : Folder path containing images
        labelFolderPath : Folder path containing label text files
        outputPath      : LMDB output path
    """
    assert os.path.exists(imageFolderPath), "Image folder does not exist"
    assert os.path.exists(labelFolderPath), "Label folder does not exist"
    
    image_files = os.listdir(imageFolderPath)
    label_files = os.listdir(labelFolderPath)
    
    assert len(image_files) == len(label_files), "Number of images and labels do not match"
    
    nSamples = len(image_files)
    env = lmdb.open(outputPath, map_size=10485742460)
    cache = {}
    cnt = 1

    for i in tqdm(range(nSamples)):
        imageFile = image_files[i]
        labelFile = label_files[i]
        
        imagePath = os.path.join(imageFolderPath, imageFile)
        labelPath = os.path.join(labelFolderPath, labelFile)
        
        assert os.path.exists(imagePath), f"Image file {imageFile} does not exist"
        assert os.path.exists(labelPath), f"Label file {labelFile} does not exist"
        
        # Read label text from file
        with open(labelPath, 'r') as f:
            label = f.read().strip()
        
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if not check_image_is_valid(imageBin):
            print('%s is not a valid image' % imagePath)
            continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples' % nSamples)

# 데이터 경로 설정
imageFolderPath = './result/newElc8'  # 이미지 파일 폴더 경로
labelFolderPath = './result/newElc8Label'  # 레이블 텍스트 파일 폴더 경로
outputPath = './val_lmdb'  # LMDB 데이터베이스 출력 경로

# LMDB 데이터베이스 생성
createDataset(imageFolderPath, labelFolderPath, outputPath)
