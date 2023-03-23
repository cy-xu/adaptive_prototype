# this file includes util functions to handle COCO format annotations
import time
import pycocotools

# https://cocodataset.org/#format-results

COCO_json_template = {
    'info': {
    'description': "demo dataset for CY & Kangyou's IAML demo",
    'version': '1.0',
    'year': time.strftime('%Y', time.localtime()),
    'date_created': time.strftime('%Y/%m/%d', time.localtime()),
    'licenses': [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}]
    },
    'images': [
    {
    "license": 6,
    "file_name": "000000481573.jpg",
    "coco_url": "http://images.cocodataset.org/val2017/000000481573.jpg",
    "height": 639,
    "width": 640,
    "date_captured": "2013-11-14 20:19:08",
    "flickr_url": "http://farm4.staticflickr.com/3586/3462542233_f8c340ec52_z.jpg",
    "id": 481573
    },
    ],
    'annotations': [
    {
    "segmentation": [[376.22,241.24,394.89,196.73,422.17,163.7,440.84,162.26,443.71,124.93,445.15,114.88,452.33,101.95,482.48,83.29,509.76,101.95,514.07,146.47,514.07,149.34,505.46,172.31,518.38,183.8,541.36,195.29,561.46,251.29,555.71,281.45,548.53,295.81,529.87,331.71,524.12,369.04,525.56,449.45,511.2,504.02,492.53,574.38,473.87,613.15,465.25,628.95,439.4,608.84,448.02,452.33,429.35,350.37,416.43,275.7]],
    "num_keypoints": 17,
    "area": 48560.56445,
    "iscrowd": 0,
    "keypoints": [472,126,2,483,118,2,463,125,2,503,132,2,449,135,2,526,208,2,428,187,2,541,282,2,375,241,1,518,335,2,408,303,1,500,350,2,423,316,1,468,510,2,397,490,1,444,623,1,386,631,1],
    "image_id": 481573,
    "bbox": [376.22,83.29,185.24,545.66],
    "category_id": 1,
    "id": 437938
    },
    ],
    'categories': [
    {
    "supercategory": "person",
    "id": 1,
    "name": "person",
    "keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
    "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    }
    ]
}