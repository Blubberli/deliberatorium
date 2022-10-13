import pandas as pd
import os
import random
from utils import read_all_maps, filter_maps_by_main_topic, filter_maps_by_depth, filter_maps_by_size
from kialo_util import remove_url_and_hashtags
from kialo_domains_util import get_maps2uniquetopic
import seaborn as sns
import matplotlib.pyplot as plt


def extract_node_samples_from_depth_bins(argument_map, node_type=None):
    """from an argument map:
    1. extract all leaf nodes (potential children to be matched to parents) that have a text longer than 5 words.
    2. filter them by PRO / CON
    3. create a dataframe
    - COL1: node objects, COL2: the depth / level of each node (in the tree), COL3: bins of each node depending on their
    tree level (3 labels); COL4: whether PRO or CON
    """
    # get all leaf nodes
    leaf_nodes = [node for node in argument_map.all_nodes if node.is_leaf]
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
    given an argument map and a dataframe with child nodes to be annotated, extract 10 candidate nodes for each child node.
    1 node of the candidates is the parent, the rest are sampled
    close relatives: sample of a maximum size of 7 is taken from nodes with a max distance of 3 (grandparents, sibling, niece, grandgrandparents)
    other candidates: random sample of all possible nodes
    """
    annotation_data = {}
    for i in range(len(target_node_df)):
        node = target_node_df["nodes"].values[i]
        coarse_level = target_node_df["coarse_level"].values[i]
        candidates = argument_map.all_nodes
        # filter for length
        # candidates = [n for n in candidates if len(n.name.split(" ")) > 5]
        close_candidates = [candidate for candidate in candidates if
                            node.shortest_path(candidate) <= 3 and node.shortest_path(candidate) > 1]
        close_candidates = [n for n in close_candidates if n != node and n != node.parent]
        other_candidates = list(set(candidates) - set(close_candidates))
        other_candidates = [n for n in other_candidates if n != node and n != node.parent]
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
        candidate_comments = annotation_frame["nodes"].values
        cleaned_comments = [remove_url_and_hashtags(str(el)) for el in candidate_comments]
        annotation_frame["cleaned_comments"] = cleaned_comments
        annotation_frame["ID"] = [n.id for n in annotation_frame["nodes"].values]
        # shuffle candidates
        annotation_frame = annotation_frame.sample(frac=1).reset_index(drop=True)
        annotation_data[node.id] = {"candidates": annotation_frame, "parent": node.parent.name,
                                    "parent.ID": node.parent.id, "target": node.name,
                                    "target_clean": remove_url_and_hashtags(node.name), "target.ID": node.id,
                                    "target.type": node.type, "target.depth": node.get_level(),
                                    "coarse.level": coarse_level}
    return annotation_data


def sample_annotation_batch_tolerant(topic, number_small_maps, number_larger_maps, output_dir, argument_maps,
                                     map2unique):
    used_maps = set()
    # extract only maps of that topic
    topic_maps = filter_maps_by_main_topic(maps=argument_maps, map2topic=map2unique, main_topic=topic)
    # make sure they are in a 'good depth range'
    filtered_maps = filter_maps_by_depth(maps=topic_maps, min_depth=5, max_depth=48)
    # make sure they have at least a size of 40 nodes
    filtered_maps = filter_maps_by_size(maps=filtered_maps, min_size=40, max_size=10000)
    # sort them by size
    sorted_maps = sorted(filtered_maps, key=lambda x: x.number_of_children())
    half = int(len(sorted_maps) / 2)
    # split them into 'smaller' and 'larger' maps
    small_maps = sorted_maps[:half]
    large_maps = sorted_maps[half:]
    print("the smallest map is %d" % small_maps[0].number_of_children())
    print("the largest map is %d" % large_maps[-1].number_of_children())
    print("number of maps left for sampling: %d" % len(sorted_maps))
    # get 8 instances from each map (total of 5 maps per topic)
    for i in range(0, number_small_maps):
        # becomes true if a map with 4 PRO and 4 CONS can be found
        sucess = False
        while not sucess:
            # get
            map = random.choice(large_maps)
            df_pro = extract_node_samples_from_depth_bins(map, node_type=1)
            # extract child nodes either from middle or specific level of the tree
            df_pro = df_pro[(df_pro['coarse_level'] == 'middle') | (df_pro['coarse_level'] == 'specific')]

            df_con = extract_node_samples_from_depth_bins(map, node_type=-1)
            df_con = df_con[(df_con['coarse_level'] == 'middle') | (df_con['coarse_level'] == 'specific')]

            if len(df_pro) >= 4 and len(df_con) >= 4 and map.id not in used_maps:
                df_pro = df_pro.sample(n=4)
                df_con = df_con.sample(n=4)
                sucess = True
                used_maps.add(map.id)
                print("%d. map of size %d and depth %d" % (i, map.number_of_children(), map.max_depth))

        target_node_df = pd.concat([df_pro, df_con])
        print("annotation frame of size %d" % len(target_node_df))

        annotation_data = extract_candidates(argument_map=map, target_node_df=target_node_df)
        write_annotation_data(output_dir, annotation_data, map)

    for i in range(0, number_larger_maps):
        sucess = False
        while not sucess:
            map = random.choice(large_maps)
            df_pro = extract_node_samples_from_depth_bins(map, node_type=1)
            df_pro = df_pro[(df_pro['coarse_level'] == 'middle') | (df_pro['coarse_level'] == 'specific')]

            df_con = extract_node_samples_from_depth_bins(map, node_type=-1)
            df_con = df_con[(df_con['coarse_level'] == 'middle') | (df_con['coarse_level'] == 'specific')]

            if len(df_pro) >= 4 and len(df_con) >= 4 and map.id not in used_maps:
                df_pro = df_pro.sample(n=4)
                df_con = df_con.sample(n=4)
                sucess = True
                used_maps.add(map.id)
                print("%d. map of size %d and depth %d" % (i, map.number_of_children(), map.max_depth))

        target_node_df = pd.concat([df_pro, df_con])
        print("annotation frame of size %d" % len(target_node_df))

        annotation_data = extract_candidates(argument_map=map, target_node_df=target_node_df)
        write_annotation_data(output_dir, annotation_data, map)


def write_annotation_data(output_dir, annotation_data, map):
    for node, annotation in annotation_data.items():
        path = os.path.join(output_dir, (node).replace(".", ""))
        os.mkdir(path)
        with open(os.path.join(path, "map_info.txt"), "w") as f:
            f.write("mapID\tclaim\tsize\tdepth\n")
            f.write("%d\t%s\t%d\t%d\n" % (map.id, map.name, map.number_of_children(), map.max_depth))
        candidate_path = os.path.join(path, "candidates.csv")
        target_node_path = os.path.join(path, "target_node.csv")
        annotation_path = os.path.join(path, "example%s.csv" % (node).replace(".", ""))

        annotation_frame = pd.DataFrame()
        annotation_frame["target"] = [remove_url_and_hashtags(annotation["target"])] + [""] * 9
        candidates = annotation["candidates"]["cleaned_comments"].values
        candidates = [str(el) for el in candidates]
        annotation_frame["candidates"] = candidates
        annotation_frame["targetID"] = [annotation["target.ID"]] * len(annotation_frame)
        annotation_frame["candidateID"] = annotation["candidates"]["ID"].values

        annotation_frame["BEST PARENT"] = [""] * len(annotation_frame)
        annotation_frame["OTHER SUITABLE PARENTS"] = [""] * len(annotation_frame)
        annotation_frame["LESS SUITABLE PARENTS"] = [""] * len(annotation_frame)
        annotation_frame.to_csv(annotation_path, sep="\t", index=False)

        annotation["candidates"].to_csv(candidate_path, sep="\t", index=False)
        with open(target_node_path, "w") as f:
            f.write("target node\ttarget node clean\tID\tparent\tparentID\ttarget depth\ttarget type\tcoarseLevel\n")
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                annotation["target"], annotation["target_clean"], annotation["target.ID"], annotation["parent"],
                annotation["parent.ID"], annotation["target.depth"],
                annotation["target.type"], annotation["coarse.level"]))


def convert_annotationdata_to_googleforms(annot_dir):
    # form questions	question type	answer start	answer end	Description
    print(annot_dir)
    output_file = open(annot_dir + "/googleformssheet.tsv", "w")
    output_file.write(
        "form questions\tquestion type\tanswer start\tanswer end\tDescription\tclaimID\ttargetID\tcandidateIDs\n")
    confidence_row = "Confidence Rating\tSCALE\t1,3,Unsure,Certain\t\tHow confident are you in your response?\t\t\t\n"
    for root, dirs, files in os.walk(annot_dir):
        for file in files:
            if "example" in file:
                example_file = os.path.join(root, file)
                map_info = pd.read_csv(example_file.replace(file, "map_info.txt"), sep="\t")
                claim = map_info["claim"].values[0]
                mapID = map_info["mapID"].values[0]
                df = pd.read_csv(example_file, sep="\t")
                target = df["target"].values[0]
                candidates = list(df["candidates"].values)
                # remove any additional line breaks in candidates and target and claim
                claim = claim.replace("\n", " ")
                target = target.replace("\n", " ")
                candidates = [el.replace("\n", " ") for el in candidates]
                targetID = df["targetID"].values[0]
                candidateIDs = list(df["candidateID"].values)
                canidate_string = "#".join(candidates)
                id_string = "##".join([str(el) for el in candidateIDs])
                selectstring = "BEST PARENT#SUITABLE PARENT#LESS SUITABLE PARENT"
                output_file.write("%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n" % (
                    target, "checkbox grid", canidate_string, selectstring, claim, mapID, targetID, id_string))
                output_file.write(confidence_row)

    output_file.close()


def create_annotation_batch():
    # extract sample
    # 1) read maps
    # 2) filter maps within a certain size range / range of depth / different topic
    # 3) return a random map
    # 4) bin the maps leaf nodes according to tree depth
    # 5) randomly sample from each bin
    # 6) for each child note:
    #   - extract all nodes within a distance of 2 and sample 8 candidates from them
    #   - extract a random sample of 2 with a distance >2
    # topics = ["gender", "economics", "immigration", "politics", "environment"]
    topics = ["environment"]
    outpath = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation200instances"
    data_path = "/Users/johannesfalk/PycharmProjects/deliberatorium/kialoV2/english"
    argument_maps = read_all_maps(data_path, debug_maps_size=1100)
    kialo2topics = "/Users/johannesfalk/PycharmProjects/deliberatorium/dataKialoV2/mapID2topicTags/ENmapID2topicTags.txt"
    main_topics = "/Users/johannesfalk/PycharmProjects/deliberatorium/dataKialoV2/maintopic2subtopic.tsv"
    map2unique, topic2submapping = get_maps2uniquetopic(kialo2topics, main_topics)
    for topic in topics:
        sample_annotation_batch_tolerant(topic=topic, argument_maps=argument_maps,
                                         output_dir="%s/%s" % (outpath, topic),
                                         number_small_maps=3,
                                         number_larger_maps=2,
                                         map2unique=map2unique)


def analyze_annotation_sample():
    root = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation200instances"
    annotated_maps = {}
    annotated_target_nodes = []
    annotated_candidates = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            df = pd.read_csv(os.path.join(path, name), sep="\t")
            if "map_info" in name:
                topic = path.split("/")[-2]
                df["topic"] = [topic] * len(df)
                annotated_maps[df["mapID"].values[0]] = df
            elif "candidates" in name:
                topic = path.split("/")[-2]
                df["topic"] = [topic] * len(df)
                annotated_candidates.append(df)
            elif "target" in name:
                topic = path.split("/")[-2]
                df["topic"] = [topic] * len(df)
                annotated_target_nodes.append(df)
    target_df = pd.concat(annotated_target_nodes)
    candidate_df = pd.concat(annotated_candidates)
    map_df = pd.concat(list(annotated_maps.values()))
    candidate_df.to_csv(os.path.join(root, "all_candidates.csv"), sep="\t", index=False)
    target_df.to_csv(os.path.join(root, "all_targets.csv"), sep="\t", index=False)
    map_df.to_csv(os.path.join(root, "all_maps.csv"), sep="\t", index=False)

def create_plots_annotation_data(map_df, target_df, candidate_df):
    sns.countplot(x="depth", data=map_df)
    sns.countplot(x="target depth", data=target_df)
    plt.show()
    # plt.show()

    map_df["quartiles"] = pd.qcut(map_df["size"].values, q=4)
    print(map_df.quartiles.values)
    newbins = [91, 226, 285, 934, 2421]
    newlabels = []
    for i in range(len(newbins) - 1):
        b = "%d-%d" % (newbins[i], newbins[i + 1])
        newlabels.append(b)
    map_df["quartiles"] = pd.cut(map_df["size"], newbins, labels=newlabels)
    sns.countplot(map_df.quartiles.values)
    plt.title("#maps: number of nodes")
    plt.show()

    candidate_df["quartiles"] = pd.qcut(candidate_df["distance"].values, q=4)
    print(candidate_df.quartiles.values)
    newbins = [2, 3, 4, 7, 26]
    newlabels = []
    for i in range(len(newbins) - 1):
        b = "%d-%d" % (newbins[i], newbins[i + 1])
        newlabels.append(b)
    candidate_df["quartiles"] = pd.cut(candidate_df["distance"], newbins, labels=newlabels)
    sns.countplot(candidate_df.quartiles.values)
    plt.title("#maps: number of nodes")
    plt.show()


if __name__ == '__main__':
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"

    topic2claims = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation_kialoV2"
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoID2MainTopic.csv"
    main_topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialo_domains.tsv"
