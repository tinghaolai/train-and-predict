## Choose

1. normal: CNN
2. lstm: CNN + lstm

## Train and run (v1)
1. record
2. Object Detection training (YOLOv8)
   * ~~personally using roboflow.com~~ (turns outs need premium to download weights)
     * add project > upload images > label manually > train model > download 
   * Recommend labelImg, open source, easy to install, and friendly gui
     * Need to switch a Yolo format
3. train
4. auto_run

## Train and run (v2, multimodal)
* record
* mark image for Yolo v8
* run `train_yolov8.py`
* run `generate_yolo_features.py`
* run `train_multimodal.py`
* run `auto_run_multimodal.py`

## Todo

* YOLOv8 + CNN
* Multi-modal Imitation Learning for training