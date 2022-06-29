# extract sample

# 1) read maps
# 2) filter maps within a certain size range / range of depth / different topic
# 3) return a random map
# 4) bin the maps leaf nodes according to tree depth
# 5) randomly sample from each bin
# 6) for each child note:
#   - extract all nodes within a distance of 2 and sample 8 candidates from them
#   - extract a random sample of 2 with a distance >2
from tqdm import tqdm
from argumentMap import KialoMap
from kialo_domains_util import get_maps2uniquetopic
from pathlib import Path
import pandas as pd
from numpy import random
import os


# create a mapping from Argument map to main topic

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
    filtered_maps = filter_maps_by_main_topic(main_topic=topic, map2topic=map2topic, maps=maps)
    filtered_maps = filter_maps_by_depth(filtered_maps, min_depth, max_depth)
    filtered_maps = filter_maps_by_size(filtered_maps, min_size, max_size)
    return random.choice(filtered_maps)


def extract_node_samples_from_depth_bins(argument_map, node_type=None):
    # get all leaf nodes
    leaf_nodes = [node for node in argument_map.all_children if node.is_leaf]
    # remove nodes that are too short
    leaf_nodes = [node for node in leaf_nodes if len(node.name.split(" ")) > 5]

    # get all levels
    levels = [node.get_level() for node in leaf_nodes]
    # get all node types
    types = [node.type for node in leaf_nodes]
    # create a data frame with three bins according to level
    df = pd.DataFrame()
    df["nodes"] = leaf_nodes
    df["levels"] = levels

    df["coarse_level"] = pd.cut(df["levels"], 3, precision=0, labels=["general", "middle", "specific"])
    df["type"] = types
    if node_type:
        df = df[df.type == node_type]

    return df


def get_informative_dataframe(nodes_list, target_node):
    """For a list of nodes return a data frame with more information about the nodes"""
    df = pd.DataFrame()
    df["nodes"] = nodes_list
    df["levels"] = [node.get_level() for node in nodes_list]
    df["node_type"] = [node.type for node in nodes_list]
    df["id"] = [node.id for node in nodes_list]
    df["distance"] = [target_node.shortest_path(node) for node in nodes_list]
    df["impact"] = [str(node.impact) for node in nodes_list]
    return df


def extract_candidates(argument_map, target_node_df):
    """
    given an argument map and a dataframe with target nodes to be annotated, extract 10 candidate nodes for each child node.
    1 node of the candidates is the parent, the rest are sampled
    close relatives: sample of a maximum size of 7 is taken from nodes with a max distance of 3 (grandparents, sibling, niece, grandgrandparents)
    other candidates: random sample of
    """
    annotation_data = {}
    for node in target_node_df["nodes"].values:
        candidates = argument_map.all_children
        # filter for length
        candidates = [node for node in candidates if len(node.name.split(" ")) > 5]
        close_candidates = [candidate for candidate in candidates if
                            node.shortest_path(candidate) <= 3]
        close_candidates = [n for n in close_candidates if n != node and n != node.parent]
        other_candidates = list(set(candidates) - set(close_candidates))
        close_candidates = get_informative_dataframe(close_candidates, node)
        other_candidates = get_informative_dataframe(other_candidates, node)
        parent_frame = get_informative_dataframe([node.parent], node)

        size_close_candidates = len(close_candidates)
        # create a sample of maximum 7 close relatives (+1 will be parent)
        if size_close_candidates > 8:
            sample_close = close_candidates.sample(7)
            sample_far = other_candidates.sample(2)
        else:
            sample_size_far = (7 - size_close_candidates) + 2
            sample_close = close_candidates
            sample_far = other_candidates.sample(sample_size_far)
        annotation_frame = pd.concat([sample_close, sample_far, parent_frame])
        # shuffle candidates
        annotation_frame = annotation_frame.sample(frac=1).reset_index(drop=True)
        annotation_data[node.id] = {"candidates": annotation_frame, "parent": node.parent.name,
                                    "parent.ID": node.parent.id, "target": node.name}
    return annotation_data


def write_annotation_data(output_dir, annotation_data, map):
    for node, annotation in annotation_data.items():
        path = os.path.join(output_dir, (node).replace(".", ""))
        os.mkdir(path)
        with open(os.path.join(path, "map_info.txt"), "w") as f:
            f.write("claim: %s size: %d depth: %d" % (map.name, map.number_of_children(), map.max_depth))
        candidate_path = os.path.join(path, "candidates.csv")
        target_node_path = os.path.join(path, "target_node.csv")
        annotation_path = os.path.join(path, "example%s.csv" % (node).replace(".", ""))

        annotation_frame = pd.DataFrame()
        annotation_frame["target"] = [annotation["target"]] + [""] * 9
        candidates = annotation["candidates"]["nodes"].values
        candidates = [str(el) for el in candidates]
        annotation_frame["candidates"] = candidates

        annotation_frame["BEST PARENT"] = [""] * len(annotation_frame)
        annotation_frame["OTHER SUITABLE PARENTS"] = [""] * len(annotation_frame)
        annotation_frame["LESS SUITABLE PARENTS"] = [""] * len(annotation_frame)
        annotation_frame.to_csv(annotation_path, sep="\t", index=False)

        annotation["candidates"].to_csv(candidate_path, sep="\t", index=False)
        with open(target_node_path, "w") as f:
            f.write("target node\tID\tparent\tparentID\n")
            f.write("%s\t%s\t%s\t%s\n" % (annotation["target"], node, annotation["parent"], annotation["parent.ID"]))


# bins kialo depth: (2-10), (10-18), (18-26), (26-34), (34-42)
# bins kialo size: (11-1000), (1000-2000), (2000-3000), (3000-4000), >4000

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path = "/Users/johannesfalk/PycharmProjects/deliberatorium/kialoV2/english"
    argument_maps = read_all_maps(data_path)
    kialo2topics = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/kialoID2MainTopic.csv"
    main_topics = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/kialo_domains.tsv"

    map2unique, topic2submapping = get_maps2uniquetopic(kialo2topics, main_topics)



    annotation_dir = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation_kialoV2/enviroment"
    # p = "/Users/johannesfalk/PycharmProjects/deliberatorium/kialo_maps/should-the-death-penalty-be-abolished-28302.txt"
    used_maps = set()
    for i in range(0, 2):
        sucess = False
        while not sucess:
            map = extract_random_map_from_filtering(maps=argument_maps, topic='enviroment', min_depth=4, max_depth=50,
                                                    min_size=200,
                                                    max_size=3000, map2topic=map2unique)

            df_pro = extract_node_samples_from_depth_bins(map, node_type=1)
            df_pro = df_pro[(df_pro['coarse_level'] == 'middle') | (df_pro['coarse_level'] == 'specific')]

            df_con = extract_node_samples_from_depth_bins(map, node_type=-1)
            df_con = df_con[(df_con['coarse_level'] == 'middle') | (df_con['coarse_level'] == 'specific')]
            if len(df_pro) >= 1 and len(df_con) >= 1 and map.id not in used_maps:
                df_pro = df_pro.sample(n=1)
                df_con = df_con.sample(n=1)
                sucess = True
                used_maps.add(map.id)
        print(map.name)

        target_node_df = pd.concat([df_con, df_pro])

        annotation_data = extract_candidates(argument_map=map, target_node_df=target_node_df)
        write_annotation_data(annotation_dir, annotation_data, map)

# human rights, enviroment, democracy, finance
