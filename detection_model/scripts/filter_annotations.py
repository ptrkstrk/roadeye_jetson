from argparse import ArgumentParser
import os
import json
from collections import Counter
import pickle
'''
The script accepts the directory with annotation files as an argument and saves their copy without the unnecessary annotations.
args:
$1 - txt with annotations files names
$2 - txt file with label names
$3 - directory with annotations
$4 - directory where we want to store new annotations
'''
def read_file_lines(path):
    text_file = open(path, "r")
    return text_file.read().splitlines()

def get_annotations(filenames, path):
    '''
    returns dictionary: key-filename, value-annotation
    '''
    annotations = {}
    for filename in filenames:
        with open(os.path.join(path, filename + '.json')) as f:
            annotations[filename] = json.load(f)
    return annotations

def filter_annotations(annots, labels, train):
    i = 0
    with open('symmetrical_signs.pkl', 'rb') as f:
        symmetrical_signs = pickle.load( f)
    for annot in annots:
        all_symmetrical = True
        annots[annot]['objects'] = [
            obj 
            for obj in annots[annot]['objects'] 
                if obj['label'] in labels
        ]

        objs_to_be_removed = []
        for obj in annots[annot]['objects']:
            if annots[annot]['width'] > 1920 and train:
                bbox = obj['bbox']
                area = (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin'])
                if (area < 150):# 300
                    #print(annots[annot]['width'])
                    i=i+1
                    objs_to_be_removed.append(obj)
            obj['label'] = labels[obj['label']]
            if(obj['label'] not in symmetrical_signs):
                all_symmetrical = False
            
        annots[annot]['all_symmetrical'] = all_symmetrical

        
        for obj in objs_to_be_removed:
            annots[annot]['objects'].remove(obj)
    print('small bboxes:', i)
    return annots

def store_annotations(annots, path):
    for filename in filenames:
        with open(os.path.join(path, filename + '.json'), 'w+') as f:
            json.dump(annots[filename], f,indent=2)
    return annotations


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("filenames", help="file with names of files with annotations")
    parser.add_argument("labels", help="file with map of traffic signs labels")
    parser.add_argument("src_dir", help="Path to directory where annotations are stored")
    parser.add_argument("target_dir", help="Path where filtered annotations will be stored")
    parser.add_argument("set_type", help="type of dataset to be filtered. Possible values: train or val")

    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    filenames=read_file_lines(args.filenames)
    annotations = get_annotations(filenames, args.src_dir)
    with open(args.labels, 'rb') as f:
        f.seek(0)
        filtered_labels = pickle.load(f)

    annotations = filter_annotations(annotations, filtered_labels, args.set_type == 'train')
    print(len(annotations))
    store_annotations(annotations, args.target_dir)