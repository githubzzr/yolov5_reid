- # YoloV8_reid

  ## 📌 Project Description
  
  This project is developed based on **Python 3.8** and **[YOLOv8](https://github.com/ultralytics/ultralytics)**. It integrates object detection and person re-identification (ReID) to support intelligent recognition and tracking in visual scenes.
  
  This repository is **forked from** [YINYIPENG-EN/yolov5_reid](https://github.com/YINYIPENG-EN/yolov5_reid) and has been **upgraded to support YOLOv8**.
  
  > ⚠️ The **model training** and **inference (recognition)** parts of the system are **completely separate**:
  >
  > - The **training pipeline** is located in the `train/` directory. You may need to **adjust some file paths** to run it properly.
  > - The **recognition system** uses the trained model weights along with YOLOv8 to perform **person re-identification**.
  > - The project  uses a **ResNet-50-IBN-a** model for ReID.
  
  > 📁 **Model weights should be placed at**:  
  > `person_search/weights/ReID_resnet50_ibn_a.pth`  
  > ⚠️ Due to GitHub file size restrictions, this file is **not included in the repository**. 
  
  ---
  
  ## 🔧 Environment Requirements
  
  - Python 3.8
  - torch >= 1.8
  - ultralytics (YOLOv8)
  - Flask 1.1.2
  
  
  
  
