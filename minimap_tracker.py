import cv2
import numpy as np
import util
import json
import torch

class MinimapTracker:
  def __init__(self, targets, model, icon_radius=11, *args, **kwargs):
    '''
    Parameters:
      targets:
          list of champion names
          irregular names are also supported, such as: 'trynd', 'kha', 'tf'...
          
      model:
          pytorch model
          
      icon_radius:
          radius of champion icons
          default: 11, if 11 doesn't work, try 12, 8
          
      map_image:
          location of minimap image
          default: 'minimap.png'
          
      map_pos:
          minimap location. tuple of ints
          if unsure, should be set to None, program will automatically detect it for you
          default: None
          
      map_size:
          minimap size. tuple of ints (width, height)
          if unsure, should be set to None, program will automatically detect it for you
          default: None
          
      map_border:
          amount of padding that will be applied to minimap.
          default: icon_radius
          
      threshold:
          a value between 0 and 1
          if it's set too high, the detection will be more precise but less frequent
          if it's set too low, the detection might be inaccuracte but more frequent
          default: 0.3
      
      show:
          used for testing.
          if True, show the ongoing tracking process on seperate windows. there has to be a cv2.waitKey() in your main loop in order for this to work
          setting this to False will boost the performance a little bit.
          default: False
    '''
    # Icon radius
    self.icon_radius = icon_radius
    
    # Classes
    with open('class_names.json','r') as f:
      self.classes = json.load(f)
    self.c2i = {j:i for i,j in enumerate(self.classes)}
    
    # Targets
    self.targets = [util.regularize(i) for i in targets] + ['Terrain']
    print(self.targets)
    self.class_indices = torch.tensor([self.c2i[i] for i in self.targets])
    self.paths = {i:{} for i in self.targets[:-1]}

    # Create classifier
    self.model = model
    self.model.eval()
    
    # Load minimap Image
    self._minimap_image_path = kwargs['map_image'] if 'map_image' in kwargs.keys() else 'minimap.png'
    self.minimap_image = cv2.imread(self._minimap_image_path)
    
    # Map position
    self.map_pos = kwargs['map_pos'] if 'map_pos' in kwargs.keys() else None
    
    # Map size
    self.map_size = kwargs['map_size'] if 'map_size' in kwargs.keys() else None
    
    # Map border (for padding)
    self.map_border = kwargs['map_border'] if 'map_border' in kwargs.keys() else self.icon_radius
    
    # Counter, increment by 1 each time track() is called.
    self.counter = 0
    
    
    self.showmap = None
    self.show = kwargs['show'] if 'show' in kwargs.keys() else False
    self.threshold =  kwargs['threshold'] if 'threshold' in kwargs.keys() else 0.2
        
  def track(self, frame, counter = None):
    assert frame is not None, 'Invalid input image'
    if counter:
      self.counter = counter
    if not self.locate_minimap(frame):
      return None    

    maparea = frame[self.map_pos[0]:self.map_pos[2],
                    self.map_pos[1]:self.map_pos[3],:]
    # Pad the maparea
    padded_map = self.pad(maparea, self.map_border)
    gray = util.grayscale(padded_map)
    circle_map = np.zeros_like(gray)
    circles = cv2.HoughCircles(image = gray,
                               method = cv2.HOUGH_GRADIENT,
                               dp = 1,
                               minDist = 8,
                               param1 = 50, # 10 to 100
                               param2 = 9.0, # 8.0 to 9.0
                               minRadius = self.icon_radius-1,
                               maxRadius = self.icon_radius+1)
    if circles is None:
      return None
    if len(circles) > 0:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
        cv2.circle(img = circle_map,
                   center = tuple(i[:2]),
                   radius = i[2]+1,
                   color = (255,255,255),
                   thickness = -1)
    circle_mask = np.uint8(circle_map / 255) * 255
    masked_map = cv2.bitwise_and(padded_map, padded_map, mask = circle_mask)
    rois = [tuple([masked_map[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]])+tuple(i[:2]) for i in circles[0,:]]
    rois = [i for i in rois if np.product(i[0].shape) > 0]
    rois = [tuple([(cv2.resize(i[0], (24, 24))/255.0).astype('float32')])+tuple(i[1:]) for i in rois]
    self.showmap = maparea.copy()
    self.classify(rois)
    self.counter += 1
    
  def classify(self, icon_list):
    self.icon_list = icon_list
    if self.show:
      cv2.imshow('icons', np.concatenate([i[0] for i in self.icon_list[:10]], axis=1))
    batch = [torch.tensor(i[0][:,:,[2,1,0]]) for i in icon_list]
    batch = torch.stack(batch)
    batch = batch.transpose(3,2).transpose(2,1)
    probs = torch.softmax(self.model(batch), dim=-1)[:, self.class_indices]
    
    best_match = probs.argmax(dim=0)
    best_match_probs = probs.max(dim=0)[0]
    coords = [icon_list[i.item()][1:] for i in best_match]
    coords = [(i[1]-self.map_border, i[0]-self.map_border) for i in coords]
    coords = [(round(x/self.map_size[0], 4), round(y/self.map_size[1], 4)) for x,y in coords]
    for i in range(len(self.targets[:-1])):
      if best_match_probs[i] > self.threshold:
        self.paths[self.targets[i]][self.counter] = coords[i]
    
    if self.show:
      try:
        for c in self.targets[:-1]:
          if self.counter in self.paths[c].keys():
            center = self.paths[c][self.counter]
            center = (int(center[1]*self.map_size[0]),#+self.map_border,
                     int(center[0]*self.map_size[1]))#+self.map_border)
            cv2.circle(self.showmap,
                       center=center,
                       radius=self.icon_radius,
                       thickness=2,
                       color=(0, 188, 188),
                       )
            cv2.putText(self.showmap,
                        text=c,
                        org=(center[0]-len(c), center[1]+2*self.icon_radius),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.35,
                        color=(0, 250, 250),
                        thickness=1,
                        )
        cv2.imshow('showmap', self.showmap)
      except:
        pass

  def locate_minimap(self, frame):
    if not self.map_pos:
      h, w = frame.shape[:2]
      resized_h = 100
      resized_w = int(resized_h / h * w)
      scaling = resized_h / h
      error = int(1/scaling)
      
      frame = util.grayscale(frame)
      resized_frame = cv2.resize(frame, (resized_w, resized_h))
      gray_minimap = util.grayscale(self.minimap_image)
      pos = [0, 0, (0, 0)]
      
      for i in range(int(resized_h*0.15), int(resized_h*0.4)):
        resized_minimap = cv2.resize(gray_minimap, (i, i))
        res = cv2.matchTemplate(resized_frame , resized_minimap, cv2.TM_CCOEFF_NORMED)
        t_pos = tuple([j.tolist()[0] for j in np.where(res == res.max())])
        if res.max() > pos[0]:
          pos = [res.max(),i, t_pos]
          
      if pos[0] >= 0.5:
        approx_y, approx_x = int(pos[2][0]/scaling), int(pos[2][1]/scaling)
        approx_mapsize = int(pos[1]/scaling)
        roi_left, roi_right = approx_x - error, approx_x + approx_mapsize + error
        roi_top, roi_bottom = approx_y - error, approx_y + approx_mapsize + error
        
        roi_left = roi_left if roi_left > 0 else 0
        roi_right = roi_right if roi_right < w else w
        roi_top = roi_top if roi_top > 0 else 0
        roi_bottom = roi_bottom if roi_bottom < h else h
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Detect second time
        pos2 = [0, 0, (0, 0)]
        for k in range(approx_mapsize-error*2, roi.shape[0]):
          resized_map = cv2.resize(gray_minimap, (k, k))
          res = cv2.matchTemplate(roi , resized_map, cv2.TM_CCOEFF_NORMED)
          t_pos = tuple([j.tolist()[0] for j in np.where(res == res.max())])
          if res.max() > pos2[0]:
            pos2 = [res.max(), k, t_pos]
        if pos2[0] > 0.5:
          self.map_pos = (roi_top + pos2[2][0], roi_left + pos2[2][1],
                          roi_top + pos2[2][0]+pos2[1], roi_left + pos2[2][1]+pos2[1])
          self.map_size = (pos2[1], pos2[1])
          return True
      return False
    else:
      return True
  
  @staticmethod
  def pad(image, padding):
    '''
        padding = (top, left, bottom, right)
        padding = all
        padding = (top_bottom, left_right)
    '''
    if type(padding) == int:
      top, left, bottom, right = padding, padding, padding, padding
    elif len(padding) == 1:
      top, left, bottom, right = padding, padding, padding, padding
    elif len(padding) == 2:
      top, left, bottom, right = padding[0], padding[1], padding[0], padding[1]
    elif len(padding) == 4:
      tpp, left, bottom, right = padding
    else:
      raise ValueError
    
    if len(image.shape) == 2:
      empty = np.zeros((image.shape[0]+top+bottom, image.shape[1]+left+right), dtype=np.uint8)
      empty[top:-bottom, left:-right] = image.copy()
    elif len(image.shape) == 3:
      empty = np.zeros((image.shape[0]+top+bottom, image.shape[1]+left+right, image.shape[2]), dtype=np.uint8)
      empty[top:-bottom, left:-right,:] = image.copy()
    return empty
  
  def save(self, savepath):
    with open(savepath, 'w') as f:
      json.dump(self.paths, f)
    print('successfully saved!')
