import os
from pathlib import Path
from argumentMap import KialoMap
import random

def read_all_maps(data_path, debug_maps_size=None):
    """read all maps from source directory and return a list of ArgMap objects"""
    maps = list(Path(data_path).glob(f"*.pkl"))
    if debug_maps_size:
        maps = sorted(maps, key=os.path.getsize)
        maps = maps[:debug_maps_size]
    argument_maps = [KialoMap(str(_map), _map.stem) for _map in tqdm(maps, f'processing maps')
                     # some maps seem to be duplicates with (1) in name
                     if '(1)' not in _map.stem]
    return argument_maps


def filter_maps_by_main_topic(main_topic, map2topic, maps):
    """provide a topic and retrieve all corresponding maps"""
    filtered_maps = [map for map in maps if int(map.id) in map2topic]
    filtered_maps = [map for map in filtered_maps if map2topic[int(map.id)] == main_topic]
    return filtered_maps


def filter_maps_by_depth(maps, min_depth, max_depth):
    """retrieve all maps from a specific depth bin, (minimum depth of tree and maximum depth)"""
    filtered_maps = [map for map in maps if map.max_depth > min_depth and map.max_depth < max_depth]
    return filtered_maps


def filter_maps_by_size(maps, min_size, max_size):
    """retrieve all maps between a minimum and maximum number of nodes"""
    filtered_maps = [map for map in maps if map.number_of_children() > min_size and map.number_of_children() < max_size]
    return filtered_maps


def extract_random_map_from_filtering(maps, min_size, max_size, min_depth, max_depth, topic, map2topic):
    """filter the list of all argument maps by node size, depth of the tree and topic.
    return a random map of that list of maps"""
    filtered_maps = filter_maps_by_main_topic(main_topic=topic, map2topic=map2topic, maps=maps)
    print("there are %d maps of the topic %s" % (len(filtered_maps), topic))
    filtered_maps = filter_maps_by_depth(filtered_maps, min_depth, max_depth)
    print("there are %d maps with between %d and %d depth" % (len(filtered_maps), min_depth, max_depth))
    filtered_maps = filter_maps_by_size(filtered_maps, min_size, max_size)
    print("there are %d maps with between %d and %d nodes" % (len(filtered_maps), min_size, max_size))
    return random.choice(filtered_maps)


def get_topic2claims(annotdir):
    topic2cclaims = {}
    for root, dirs, files in os.walk(annotdir):
        for file in files:
            if "example" in file:
                example_file = os.path.join(root, file)
                map_info = example_file.replace(file, "map_info.txt")
                claim = open(map_info).readlines()[0].split("size")[0].strip()
                topic = map_info.split("/")[-3]
                if topic not in topic2cclaims:
                    topic2cclaims[topic] = set()
                topic2cclaims[topic].add(claim)
    return topic2cclaims
