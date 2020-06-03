import cv2
import numpy as np
import json
import os

def grayscale(img):
    '''
    convert image to grayscale
    parameters:
        img: numpy array with shape = (X, Y, C)
    return:
        numpy array with shape = (X, Y)
    '''
#    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return weighted_grayscale(img, 0.15, 0.2, 0.65)

def weighted_grayscale(img, alpha, beta, gamma):
    return cv2.addWeighted(cv2.addWeighted(img[:,:,0], alpha, img[:,:,1], beta, 0), 1.0, img[:,:,2], gamma, 0)
    
def load_yt(url):
    '''
    load video stream from Youtube
    parameters:
        url: str, URL or 11 digit VideoID of the video
    return:
        pafy.Stream object
    '''
    import pafy
    print(f'Start loading from {url}')
    vpafy = pafy.new(url)
    best = vpafy.getbestvideo("any", False)
    print('Title:',best.title)
    print('Filesize:',round(best.get_filesize()/ (1000 * 1000), 2), 'MB')
    print('Resolution:',best.resolution)
    return best

def HSV(h, s, v):
    # (360, 100%, 100%) => (180, 255, 255)
    return int(h/2), int(s/100 * 255), int(v/100*255)
    
champ_names = json.loads(open('champ_nicknames.json', 'r').read())

def regularize(name):
    name = list(name)
    name[0] = name[0].upper()
    name = ''.join(name)
    for k,  v in champ_names.items():
        if name in v:
            return k
    return name

def iterate_folder(path):
    if os.path.exists(path):
        folders = os.walk(path)
        folders = [(i[0].replace('\\','/'),i[2]) for i in folders if i[2]]
        pics = [ i[0]+'/'+j for i in folders for j in i[1]]
        return pics
    else:
        print("Path doesn't exist")
        return []
    
def labeled_files(folder ,label):
    files = [ (label, folder+'/'+i) for i in list(os.walk(folder))[0][2]]
    return files
    
