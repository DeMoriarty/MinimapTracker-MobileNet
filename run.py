import cv2
import time
from minimap_tracker import MinimapTracker
from model import mobilenet_v3

video_path = "videos/grf.mp4"
#targets = ["Jayce","Rek'Sai","Lissandra","Ezreal","Karma", 
#           "Ornn","Jarvan IV","Morgana","Lucian","Shen"]
targets = ["jayce","reksai","liss","ez","karma", 
           "ornn","j4","morg","lucian","shen"]
starts_at = 97 * 30
icon_radius=8
show = True

### You need to have youtube-dl and pafy library installed
#video_path = util.load_yt('https://www.youtube.com/watch?v=LST3AF-bpIA').url
#targets = ["Kennen","Jarvan IV","Sejuani","Ezreal", "Nautilus", 
#           "Aatrox","Lee Sin","Taric","Miss Fortune","LeBlanc"]
#starts_at = 83 * 30
#icon_radius=11

cap = cv2.VideoCapture(video_path)
model = mobilenet_v3(version=3)
tracker = MinimapTracker(targets=targets,
                         model=model,
                         icon_radius=icon_radius,
                         show=show,
                         )
counter = -1
time_q = [0] * 30
try:
  while cap.isOpened():
    tm = time.time()
    counter += 1
    ok, frame = cap.read()
    if counter <= starts_at:
      continue
    if not ok or frame is None:
      break
    tracker.track(frame, counter)
    time_q = time_q[1:] + [time.time()-tm]
    if counter % 90 == 0:
      print('frame:', counter,'FPS:', 30/sum(time_q))
      
    if show:
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        break
except Exception as E:
  print(E)
finally:
  tracker.save('paths/grf.json')
  cap.release()
  cv2.destroyAllWindows()