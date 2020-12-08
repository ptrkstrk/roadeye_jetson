import os
import json
import numpy as np
from detectron2.structures import BoxMode
import pickle

def read_list_from_file(path):
    text_file = open(path, "r")
    return text_file.read().splitlines()


def get_mapillary_data(num_train=52, num_val=500, num__test=500):
    train = read_list_from_file('/mnt/CommonData/dataset/splits/train.txt')
    val = read_list_from_file('/mnt/CommonData/dataset/splits/val.txt')
    test_names = read_list_from_file('/mnt/CommonData/dataset/splits/test.txt')
    

def get_annotations(img_ids, path):
    """
    returns dictionary: key:img id, value:json annotation
    """
    annotations = {}
    for id in img_ids:
        with open(os.path.join(path, id + '.json')) as f:
            annotations[id] = json.load(f)
    return annotations

        

def prepare_MTSD_for_detectron(ids_file, annots_dir, img_dir, labels_ids_file):
    """ 
    function returns MTSD dataset in format required by detectron2 
    (https://detectron2.readthedocs.io/tutorials/datasets.html) 
  
    Parameters: 
    ids_file (string): file with list of images ids
    annots_dir (string): directory where annotations are stored
    img_dir (string): directory where images are stored
    labels_ids_file (string): .pkl file with dictionary of labels and their ids
    Returns: 
    int: dataset dictionary that matches detectron2 requirements
  
    """
    img_ids = read_list_from_file(ids_file)
    annotations = get_annotations(img_ids, annots_dir)

    with open(labels_ids_file, 'rb') as f:
        lbl_id_map = pickle.load(f)

    dataset_dicts = []
    for im_id in img_ids:
        record = {}
        img_filename = os.path.join(img_dir, im_id + ".jpg")
        annot_filename = os.path.join(annots_dir, im_id + ".json")   
            
        record["file_name"] = img_filename
        record["image_id"] = im_id
        record["height"] = annotations[im_id]['height']
        record["width"] = annotations[im_id]['width']
        record["all_symmetrical"] = annotations[im_id]["all_symmetrical"]
        
        objs = []
        for obj in annotations[im_id]['objects']:

            bbox = obj['bbox']
            det2_obj = {
                    "bbox": [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    #"segmentation": [poly],
                    "category_id": lbl_id_map[obj['label']],
                }
            objs.append(det2_obj)

        record["annotations"] = objs
        #remove empty annotations
        if(len(objs) != 0):
            dataset_dicts.append(record)
    return dataset_dicts


def convert_MTSD_to_COCO(ids_file, annots_dir, img_dir, labels_ids_file):
    """ 
    function returns MTSD dataset in COCO format
  
    Parameters: 
    ids_file (string): file with list of images ids
    annots_dir (string): directory where annotations are stored
    img_dir (string): directory where images are stored
    labels_ids_file (string): .pkl file with dictionary of labels and their ids
    Returns: 
    int: dataset dictionary that matches detectron2 requirements


    output = {'info':"Annotations for MTSD dataset converted to COCO format with filtered labels"
              'images': [],
              'categories': [],
              'licenses': [],
              'annotations': []} # Prepare output
  
    """
    img_ids = read_list_from_file(ids_file)
    annotations = get_annotations(img_ids, annots_dir)

    with open(labels_ids_file, 'rb') as f:
        lbl_id_map = pickle.load(f)

    dataset_dict = {'info':"Annotations for MTSD dataset converted to COCO format with filtered labels",
              'images': [],
              'categories': [],
              'licenses': [],
              'annotations': []}

    #adding labels with their ids
    for lbl in lbl_id_map:
        cat_record = {}
        cat_record['id'] = lbl_id_map[lbl]
        cat_record['name'] = lbl
        dataset_dict['categories'].append(cat_record)

    annot_id =0
    for num_id, id in enumerate(img_ids):
        if id == 'a':
            print(id)
        img_record = {}
        img_filename = os.path.join(img_dir, id + ".jpg")
        annot_filename = os.path.join(annots_dir, id + ".json")   
        
        img_record['id'] = num_id
        img_record["width"] = annotations[id]['width']
        img_record["height"] = annotations[id]['height']
        img_record["file_name"] = img_filename
        dataset_dict['images'].append(img_record)

        for obj in annotations[id]['objects']:
            annot_id = annot_id+1
            detection_record = {}
            detection_record['id'] = annot_id
            detection_record['image_id'] = num_id
            
            bbox = obj['bbox']
            w = bbox['xmax'] - bbox['xmin']
            h = bbox['ymax'] - bbox['ymin']
            detection_record['bbox'] = [bbox['xmin'], bbox['ymin'], w, h]
            detection_record['area'] = w * h
            detection_record['iscrowd'] = 0
            detection_record['category_id'] = lbl_id_map[obj['label']]
            dataset_dict['annotations'].append(detection_record)


    return dataset_dict


def write_json(data, filepath):
    """Write JSON file"""
    dir_ = os.path.split(filepath)[0]
    assert os.path.isdir(dir_), "Directory %s does not exist" % dir_

    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)
