![](https://github.com/DeMoriarty/MinimapTracker-MobileNet/blob/master/tracked.png)   

# MinimapTracker-MobileNet
This is a minimap tracker for League of Legends.   
## Dependency
this code depends on OpenCV and PyTorch.
please follow the instructions on official OpenCV and PyTorch website

## How to use it?
### 1. download the repository
```
$ git clone https://github.com/DeMoriarty/MinimapTracker-MobileNet.git
```
### 2. edit run.py
### 3. change "video_path" to path of your video file
```python
video_path = "videos/lck.mp4"  
```
it's also possible to load videos directly from youtube, but you need to have pafy library installed 
```
$ pip install pafy
```
```python
video_path = util.load_yt("https://www.youtube.com/watch?v=LST3AF-bpIA").url  
```
### 4. change "save_path" to where you want to save the output JSON file
```python
save_path = "paths/lck.json"
```
### 4. change "targets" to the name/nickname of champions you want to track
```python
targets = ["morgana", "j4", "tf"]
```
### 5. optional: change "starts_at" to numbers of frames you want to skip before start tracking
this is usefull when there is loading screen at the start of the video.  
let's say we want to skip to 1:34, then 
```python
starts_at = (minutes * 60 + seconds) * frame_rate = (60 * 1 + 34) * 30
```
### 6. optional: change "icon_radius"
usually icon_radius should be set to 11, but if this doesn't work, you can try other values under 20.  
### 7. optional: set "show" to True if you want to see the tracking process
### 8. run the script
```
$ python run.py
```
