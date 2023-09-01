# Detection and segmentation of a self-collected Pyraminx Dataset

## Pyraminx Dataset

### Pyraminx photographing
I chose a Pyraminx as the object for detection and segmentation. It is a bright triangular object, each side of it has a different color. I took photos of it from different angles, in different places, in the light and in the shade.

In addition, I tried to rotate Pyraminx vertices, so that the figure became less like a triangle. I supposed it would make the task more difficult.

### Labeling

The task focuses on object detection and segmentation, so I marked the boundaries using Roboflow's built-in Smart Polygon tool. This instrument helped speed up the process and made the results quite accurate. Sometimes I helped the tool by changing the selection by hands.

Example of the labeled image
<div align="center">
  <img
    width="640"
    src="http://drive.google.com/uc?export=view&id=1LEIDgeFjzcjFJwBESynUERoJe_ALfi7Y"
  >
</div>


### Preprocessing
As preprocessing steps I used auto-orient and resize transformations suggested by the roboflow.

### Augmentation

To increase the dataset size, I used 8 cutouts with the size 10%. From 108 images it became of size 254.


In this work, I fine-tuned 4 different models on "Pyraminx Dataset": Mask RCNN, Faster RCNN, Yolo v4 and Yolo v8. While creating this work, I used the tutorials* presented in the labs classes. To train and test models I used Tesla T4 GPU on Google.Colab.  

### 1. Mask RCNN.
Among all the models used in this paper, Mask RCNN is the only one that solves the instance segmentation problem. It is based on the principle of first finding a bounding box and then segmenting the objects inside each box. Therefore, I will discuss the segmentation results separately, and for comparison with other models, I will use the metrics from the bounding box detection.

I used a COCO-pretrained R50-FPN Mask R-CNN model from Detectron2 zoo.
It was run for 298 iterations 0:01:39 (training speed 0.3348 s/iter) with inference 0.0786 s/iter.

Evaluation results for segmentation:

|   AP   |  AP50  |  AP75  |
|:------:|:------:|:------:|
| 68.395 | 79.307 | 79.307 |

For detection metrics are a bit lesser: mAP is around 55%. The low average precision may be because the model contains 50 million parameters - a big number for which (even pretrained) "Pyraminx dataset" is not enough.

### 2. Faster RCNN

Similarly to previous model, I used a COCO-pretrained R50-FPN Faster R-CNN model from Detectron2 zoo.

|Model | Training Speed, s/iter | Training time, s |Inference, s/iter|   AP, %   |  AP50, %  |  AP75, %  |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Faster RCNN| 0.3148|93| 0.1595 | 52.515 | 84.082 | 56.723 |
|Mask RCNN| 0.3348|99|0.0786 | 68.395 | 79.307 | 79.307 |

The comparison shows that Faster RCNN speed is near to Mask RCNN. We should note that Mask RCNN uses Faster RCNN to get the bounding boxes of objects, so complexity of the second model is lower. Also, their max_mem equal to 1941M. Average precision metrics differ, but not in a big range.

### 3. YOLO v4

I used YOLO v4 Tiny from Darknet framework. Similar to previous model, it solves object detection problem. YOLO v4 has 38 layers, and its training takes much more time comparing to other models considered here. Hewever, YOLO v4 gives the predictions better than Faster RCNN and bounding boxes of Mask RCNN.

### 4. YOLO v8

Another YOLO that I used was of verion 8: Ultralytics YOLOv8.0.20. It contains 225 layers, around 11 million parameters and 28.4 GFLOPs with train inference of 8.5 ms per image and test - 21 ms.

|Model | Training Speed, s/iter | Training time, min | Number of Iterations |   AP50, %   | GFLOPs | Number of layers |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|YOLO v4| 0.89 | 30 | 2000 | 95.24 | 6.787 | 38 |
|YOLO v8| 0.3 | 6 | 25 | 99 |28.4| 225 |

This comparison table shows that YOLO v8 is winning YOLO v4 in precision and speed.

As a result of comparing the models, I concluded that the pyraminx is a difficult object for both detection and segmentation. The photos with rotated corner or with only part of the object made the task even harder. Most probably, the dataset is small for this task and adding more images may improve the performance of the models, as well as better tuning models parameters.

Based on the results one could conclude that YOLO v8 performed better than other considered models. It had the highest AP scores and learned fast enough, but slower than Faster RCNN and Mask RCNN. In addition, YOLO v8 was the most convenient to use - the implementation of the solution was fast and easy, which is not true for YOLO v4, for example. However, it is worth mentioning that increasing the dataset size may change the obtained conclusion, and then more experiments should be conducted.


#### Used sources:
* Lectures and labs of AML course
* roboflow.com
* https://github.com/facebookresearch/detectron2
* https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
* https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg
* https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb

