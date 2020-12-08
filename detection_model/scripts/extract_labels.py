from argparse import ArgumentParser
import os
import json
from collections import Counter
import pickle 

'''
The script takes a directory with annotation files as a argument and extracts from them label names, 
filters them and saves the file with the map of labels. The map is used to filter out the unused signs 
and group label(e.g. different airport signs into the same sign label).
args:
$1 - txt file with names of annotation files
$2 - directory with annotations
$3 - file where we want to store the labels map
'''

def read_filenames(path):
    text_file = open(path, "r")
    return text_file.read().splitlines()

#returns dictionary: key-filename, value-annotation
def get_annotations(filenames, path):
    annotations = {}
    for filename in filenames:
        with open(os.path.join(path, filename + '.json')) as f:
            annotations[filename] = json.load(f)
    return annotations

def get_labels(annots):
    labels = set()
    for img in annots:
        for obj in annots[img]['objects']:
            labels.add(obj['label'])
    return labels

def get_required_labels(annots):
    labels = set(get_labels(annots))
    #usuwam znaki których nie uzywam
    labels = [ l for l in labels if not l.startswith('complementary') ]
    labels.remove("information--highway-interstate-route--g2")
    labels.remove("information--parking--g3")
    labels.remove("information--tram-bus-stop--g2")
    labels.remove("other-sign")#!!!!
    labels.remove("regulatory--keep-left--g2")
    labels.remove("regulatory--lane-control--g1")
    labels.remove("regulatory--left-turn-yield-on-green--g1")
    labels.remove("regulatory--maximum-speed-limit-100--g3")
    labels.remove("regulatory--maximum-speed-limit-25--g2")
    labels.remove("regulatory--maximum-speed-limit-30--g3")
    labels.remove("regulatory--maximum-speed-limit-35--g2")
    labels.remove("regulatory--maximum-speed-limit-40--g3")
    labels.remove("regulatory--maximum-speed-limit-45--g3")
    labels.remove("regulatory--maximum-speed-limit-55--g2")
    labels.remove("regulatory--no-stopping--g5")
    labels.remove("regulatory--no-turn-on-red--g1")
    labels.remove("regulatory--no-turn-on-red--g2")
    labels.remove("regulatory--no-turn-on-red--g3")
    labels.remove("regulatory--one-way-left--g2")
    labels.remove("regulatory--one-way-right--g2")
    labels.remove("regulatory--one-way-right--g3")
    labels.remove("regulatory--parking-restrictions--g2")
    labels.remove("regulatory--pass-on-either-side--g2")
    labels.remove("regulatory--passing-lane-ahead--g1")
    labels.remove("regulatory--road-closed--g2")
    labels.remove("regulatory--stop--g2")
    labels.remove("regulatory--stop-here-on-red-or-flashing-light--g1")
    labels.remove("regulatory--stop-here-on-red-or-flashing-light--g2")
    labels.remove("regulatory--turn-left--g3")
    labels.remove("regulatory--turn-right--g3")
    labels.remove("warning--pass-left-or-right--g2")
    labels.remove("warning--pedestrians-crossing--g10")
    labels.remove("warning--pedestrians-crossing--g12")
    labels.remove("warning--road-narrows--g2")
    labels.remove("warning--road-narrows-right--g2")
    labels.remove("warning--road-narrows-left--g2")
    labels.remove("warning--road-widens--g1")
    labels.remove("warning--road-widens-right--g1")
    labels.remove("regulatory--no-pedestrians--g2")
    labels.remove("warning--divided-highway-ends--g2")
    labels.remove("information--motorway--g1")
    labels.remove("regulatory--turn-left--g2")
    labels.remove("regulatory--turn-right--g2")
    labels.remove("warning--roundabout--g2")
    labels.remove("regulatory--weight-limit--g1")
    labels.remove("warning--uneven-road--g1")
    labels.remove("information--gas-station--g1")
    labels.remove("information--gas-station--g3")
    labels.remove("information--hospital--g1")
    labels.remove("regulatory--no-motor-vehicles-except-motorcycles--g2")

    labels.remove("warning--junction-with-a-side-road-acute-right--g1")
    labels.remove("warning--junction-with-a-side-road-acute-left--g1")
    labels.remove("warning--junction-with-a-side-road-perpendicular-left--g1")
    labels.remove("warning--junction-with-a-side-road-perpendicular-right--g1")

    #<100 wystapien
    labels.remove("information--bike-route--g1")
    labels.remove("information--bus-stop--g1")
    labels.remove("information--children--g1")
    labels.remove("information--emergency-facility--g2")
    labels.remove("information--food--g2")
    labels.remove("information--interstate-route--g1")
    labels.remove("information--limited-access-road--g1")
    labels.remove("information--parking--g2")
    labels.remove("information--parking--g6")
    labels.remove("information--pedestrians-crossing--g2")
    labels.remove("information--road-bump--g1")
    labels.remove("information--safety-area--g2")
    labels.remove("information--telephone--g1")
    labels.remove("information--telephone--g2")
    labels.remove("information--trailer-camping--g1")
    labels.remove("regulatory--bicycles-only--g2")
    labels.remove("regulatory--bicycles-only--g3")
    labels.remove("regulatory--buses-only--g1")
    labels.remove("regulatory--do-not-block-intersection--g1")
    labels.remove("regulatory--dual-path-bicycles-and-pedestrians--g1")
    labels.remove("regulatory--end-of-bicycles-only--g1")
    labels.remove("regulatory--end-of-no-parking--g1")
    labels.remove("regulatory--end-of-prohibition--g1")
    labels.remove("regulatory--give-way-to-oncoming-traffic--g1")
    labels.remove("regulatory--go-straight--g3")
    labels.remove("regulatory--go-straight-or-turn-left--g2")
    labels.remove("regulatory--go-straight-or-turn-left--g3")
    labels.remove("regulatory--go-straight-or-turn-right--g3")
    labels.remove("regulatory--keep-right--g2")
    labels.remove("regulatory--keep-right--g6")
    labels.remove("regulatory--no-bicycles--g1")
    labels.remove("regulatory--no-bicycles--g2")
    labels.remove("regulatory--no-buses--g3")
    labels.remove("regulatory--no-motor-vehicles-except-motorcycles--g1")
    labels.remove("regulatory--no-motorcycles--g1")
    labels.remove("regulatory--no-motorcycles--g2")
    labels.remove("regulatory--no-overtaking--g4")
    labels.remove("regulatory--no-overtaking-by-heavy-goods-vehicles--g1")
    labels.remove("regulatory--no-parking--g6")
    labels.remove("regulatory--no-parking-or-no-stopping--g1")
    labels.remove("regulatory--no-parking-or-no-stopping--g2")
    labels.remove("regulatory--no-parking-or-no-stopping--g3")
    labels.remove("regulatory--no-parking-or-no-stopping--g5")
    labels.remove("regulatory--no-pedestrians--g1")
    labels.remove("regulatory--no-straight-through--g1")
    labels.remove("regulatory--no-u-turn--g2")
    labels.remove("regulatory--no-u-turn--g3")
    labels.remove("regulatory--pedestrians-only--g2")
    labels.remove("regulatory--radar-enforced--g1")
    labels.remove("regulatory--road-closed-to-vehicles--g1")
    labels.remove("regulatory--shared-path-bicycles-and-pedestrians--g1")
    labels.remove("regulatory--stop--g10")
    labels.remove("regulatory--stop-signals--g1")
    labels.remove("regulatory--triple-lanes-turn-left-center-lane--g1")
    labels.remove("regulatory--u-turn--g1")
    labels.remove("regulatory--wrong-way--g1")
    labels.remove("warning--added-lane-right--g1")
    labels.remove("warning--bicycles-crossing--g1")
    labels.remove("warning--bicycles-crossing--g2")
    labels.remove("warning--children--g2")
    labels.remove("warning--crossroads-with-priority-to-the-right--g1")
    labels.remove("warning--divided-highway-ends--g1")
    labels.remove("warning--domestic-animals--g1")
    labels.remove("warning--double-curve-first-left--g2")
    labels.remove("warning--double-curve-first-right--g2")
    labels.remove("warning--double-reverse-curve-right--g1")
    labels.remove("warning--double-turn-first-right--g1")
    labels.remove("warning--dual-lanes-right-turn-or-go-straight--g1")
    labels.remove("warning--emergency-vehicles--g1")
    labels.remove("warning--falling-rocks-or-debris-right--g2")
    labels.remove("warning--flaggers-in-road--g1")
    labels.remove("warning--horizontal-alignment-left--g1")
    labels.remove("warning--horizontal-alignment-right--g1")
    labels.remove("warning--narrow-bridge--g3")
    labels.remove("warning--railroad-crossing--g1")
    labels.remove("warning--railroad-crossing--g3")
    labels.remove("warning--railroad-crossing-with-barriers--g2")
    labels.remove("warning--railroad-crossing-without-barriers--g1")
    labels.remove("warning--railroad-intersection--g3")
    labels.remove("warning--roadworks--g2")
    labels.remove("warning--roadworks--g3")
    labels.remove("warning--roadworks--g4")
    labels.remove("warning--roadworks--g6")
    labels.remove("warning--steep-descent--g2")
    labels.remove("warning--t-roads--g2")
    labels.remove("warning--traffic-merges-left--g1")
    labels.remove("warning--traffic-signals--g2")
    labels.remove("warning--trail-crossing--g2")
    labels.remove("warning--two-way-traffic--g1")
    labels.remove("warning--two-way-traffic--g2")
    labels.remove("warning--wild-animals--g1")
    labels.remove("warning--winding-road-first-right--g3")
    labels.remove("warning--y-roads--g1")
    labels.remove("regulatory--dual-lanes-go-straight-on-left--g1")
    labels.remove("regulatory--dual-lanes-go-straight-on-right--g1")
    labels.remove("regulatory--no-vehicles-carrying-dangerous-goods--g1")
    labels.remove("regulatory--no-parking--g2") 
    labels.remove("warning--traffic-merges-right--g2")


    #łączenie znaków w grupy
    filtered_lbls_map ={e:e for i,e in enumerate(labels)}
    filtered_lbls_map['information--airport--g2'] = 'information--airport--g1'
    filtered_lbls_map['regulatory--no-heavy-goods-vehicles--g4'] = 'regulatory--no-heavy-goods-vehicles--g2'
    filtered_lbls_map['regulatory--no-heavy-goods-vehicles--g5'] = 'regulatory--no-heavy-goods-vehicles--g2'
    filtered_lbls_map['warning--road-narrows-right--g1'] = 'warning--road-narrows--g1'
    filtered_lbls_map['warning--road-narrows-left--g1'] = 'warning--road-narrows--g1'
    filtered_lbls_map['warning--texts--g3'] = 'warning--texts--g1'
    filtered_lbls_map['warning--texts--g2'] = 'warning--texts--g1'
    filtered_lbls_map['information--parking--g5'] = 'information--parking--g1'
    filtered_lbls_map['regulatory--maximum-speed-limit-led-100--g1'] = 'regulatory--maximum-speed-limit-led--g1'
    filtered_lbls_map['regulatory--maximum-speed-limit-led-80--g1'] = 'regulatory--maximum-speed-limit-led--g1'
    filtered_lbls_map['regulatory--maximum-speed-limit-led-60--g1'] = 'regulatory--maximum-speed-limit-led--g1'
    filtered_lbls_map['regulatory--one-way-left--g3'] = "regulatory--one-way-left--g1"
    return filtered_lbls_map


def store_labels(labels, path):
    with open(path, 'wb') as f:
        pickle.dump(labels, f)
    #with open(path, 'w+') as file:
    #    file.writelines("%s\n" % l for l in sorted(labels))

def map_labels_to_ids(labels, path):
    lbl_ids = {}
    i = 0
    print(len(set(labels.values())))
    for l in sorted(set(labels.values())):
        lbl_ids[l]=i
        i+=1
    with open(os.path.join(path), "wb") as file:
        pickle.dump(lbl_ids, file)

def map_ids_to_labels(lbl_id_map, path):
    id_lbls = {}
    i = 0
    for l in lbl_id_map:
        id_lbls[str(lbl_id_map[l])]=l
        i+=1
    with open(os.path.join(path), "wb") as file:
        pickle.dump(id_lbls, file)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("filenames", help="Path to file with names of files with annotations")
    parser.add_argument("src_dir", help="Path to directory where annotations are stored")
    parser.add_argument("target_labels_file", help="Path to file where map of filtered labels will be stored")
    parser.add_argument("target_dict_lblid_file", help="Path to file where dict of filtered labels with their ids will be stored. Must be pickle file")
    parser.add_argument("target_dict_idlbl_file", help="Path to file where dict of ids with labels will be stored. Must be pickle file")

    args = parser.parse_args()

    filenames=read_filenames(args.filenames)
    annotations = get_annotations(filenames, args.src_dir)
    filtered_labels = get_required_labels(annotations)
    store_labels(filtered_labels, args.target_labels_file)
    map_labels_to_ids(filtered_labels, args.target_dict_lblid_file)
    
    with open(args.target_dict_lblid_file, 'rb') as f:
        lbl_id_map = pickle.load(f)

    map_ids_to_labels(lbl_id_map, args.target_dict_idlbl_file)