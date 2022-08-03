# extract sample

# 1) read maps
# 2) filter maps within a certain size range / range of depth / different topic
# 3) return a random map
# 4) bin the maps leaf nodes according to tree depth
# 5) randomly sample from each bin
# 6) for each child note:
#   - extract all nodes within a distance of 2 and sample 8 candidates from them
#   - extract a random sample of 2 with a distance >2
import numpy as np
from tqdm import tqdm
from argumentMap import KialoMap
from kialo_domains_util import get_maps2uniquetopic
from pathlib import Path
import pandas as pd
from numpy import random
import os
from skll.metrics import kappa, correlation
from sklearn.metrics import f1_score, classification_report
import seaborn as sns
import glob
from evaluation import Evaluation as eval
from kialo_util import remove_url_and_hashtags


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
    print(main_topic)
    print(map2topic)
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
    print("there are %d maps of the topic %s" % (len(filtered_maps), topic))
    filtered_maps = filter_maps_by_depth(filtered_maps, min_depth, max_depth)
    print("there are %d maps with between %d and %d depth" % (len(filtered_maps), min_depth, max_depth))
    filtered_maps = filter_maps_by_size(filtered_maps, min_size, max_size)
    print("there are %d maps with between %d and %d nodes" % (len(filtered_maps), min_size, max_size))
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


# general target node: (high in the tree)
# more specific target node: (low in the tree)

# other candidates: 7 close ones (distance <=3), 3 far ones (distance >3)

# problem bot always so many

# remove the links
#

def extract_candidates(argument_map, target_node_df):
    """
    given an argument map and a dataframe with target nodes to be annotated, extract 10 candidate nodes for each child node.
    1 node of the candidates is the parent, the rest are sampled
    close relatives: sample of a maximum size of 7 is taken from nodes with a max distance of 3 (grandparents, sibling, niece, grandgrandparents)
    other candidates: random sample of
    """
    annotation_data = {}
    for i in range(len(target_node_df)):
        node = target_node_df["nodes"].values[i]
        coarse_level = target_node_df["coarse_level"].values[i]
        candidates = argument_map.all_children
        # filter for length
        # candidates = [n for n in candidates if len(n.name.split(" ")) > 5]
        close_candidates = [candidate for candidate in candidates if
                            node.shortest_path(candidate) <= 3 and node.shortest_path(candidate) > 1]
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
    small_maps = sorted_maps[:half]
    large_maps = sorted_maps[half:]
    print("the smallest map is %d" % small_maps[0].number_of_children())
    print("the largest map is %d" % large_maps[-1].number_of_children())
    print("number of maps left for sampling: %d" % len(sorted_maps))

    for i in range(0, number_small_maps):
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


def sample_annotation_batch(topic, number_small_maps, number_larger_maps, output_dir, argument_maps, map2unique):
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
    small_maps = sorted_maps[:half]
    large_maps = sorted_maps[half:]
    print("the smallest map is %d" % small_maps[0].number_of_children())
    print("the largest map is %d" % large_maps[-1].number_of_children())
    print("number of maps left for sampling: %d" % len(sorted_maps))

    for i in range(0, number_small_maps):
        sucess = False
        while not sucess:
            map = random.choice(small_maps)
            df_pro = extract_node_samples_from_depth_bins(map, node_type=1)
            df_pro_middle = df_pro[(df_pro['coarse_level'] == 'middle')]
            df_pro_specific = df_pro[(df_pro['coarse_level'] == 'specific')]

            df_con = extract_node_samples_from_depth_bins(map, node_type=-1)
            df_con_middle = df_con[(df_con['coarse_level'] == 'middle')]
            df_con_specific = df_con[(df_con['coarse_level'] == 'specific')]
            print(len(df_con_middle), len(df_pro_middle), len(df_con_specific), len(df_pro_specific))
            if len(df_pro_middle) >= 2 and len(df_pro_specific) >= 2 and len(df_con_middle) >= 2 and len(
                    df_con_specific) >= 2 and map.id not in used_maps:
                df_pro_middle = df_pro_middle.sample(n=2)
                df_pro_specific = df_pro_specific.sample(n=2)

                df_con_middle = df_con_middle.sample(n=2)
                df_con_specific = df_con_specific.sample(n=2)
                sucess = True
                used_maps.add(map.id)
                print("%d. map of size %d and depth %d" % (i, map.number_of_children(), map.max_depth))

        target_node_df = pd.concat([df_pro_middle, df_pro_specific, df_con_middle, df_con_specific])
        print("annotation frame of size %d" % len(target_node_df))

        annotation_data = extract_candidates(argument_map=map, target_node_df=target_node_df)
        write_annotation_data(output_dir, annotation_data, map)
    for i in range(0, number_larger_maps):
        sucess = False
        while not sucess:
            map = random.choice(large_maps)
            df_pro = extract_node_samples_from_depth_bins(map, node_type=1)
            df_pro_middle = df_pro[(df_pro['coarse_level'] == 'middle')]
            df_pro_specific = df_pro[(df_pro['coarse_level'] == 'specific')]

            df_con = extract_node_samples_from_depth_bins(map, node_type=-1)
            df_con_middle = df_con[(df_con['coarse_level'] == 'middle')]
            df_con_specific = df_con[(df_con['coarse_level'] == 'specific')]
            print(len(df_con_middle), len(df_pro_middle), len(df_con_specific), len(df_pro_specific))

            if len(df_pro_middle) >= 2 and len(df_pro_specific) >= 2 and len(df_con_middle) >= 2 and len(
                    df_con_specific) >= 2 and map.id not in used_maps:
                df_pro_middle = df_pro_middle.sample(n=2)
                df_pro_specific = df_pro_specific.sample(n=2)

                df_con_middle = df_con_middle.sample(n=2)
                df_con_specific = df_con_specific.sample(n=2)
                sucess = True
                used_maps.add(map.id)
                print("%d. map of size %d and depth %d" % (i, map.number_of_children(), map.max_depth))

        target_node_df = pd.concat([df_pro_middle, df_pro_specific, df_con_middle, df_con_specific])
        print("annotation frame of size %d" % len(target_node_df))

        annotation_data = extract_candidates(argument_map=map, target_node_df=target_node_df)
        write_annotation_data(output_dir, annotation_data, map)


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


# bins kialo depth: (2-10), (10-18), (18-26), (26-34), (34-42)
# bins kialo size: (11-1000), (1000-2000), (2000-3000), (3000-4000), >4000

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
    for topic, claims in topic2cclaims.items():
        print(topic, claims)


def merge_googleform_answers(answer_dir, max_documents):
    answers = []
    for i in range(1, max_documents + 1):
        answersheet = pd.read_csv("%s/KialoAnnotations%d.csv" % (answer_dir, i), sep=",")
        answers.append(answersheet)
    df = pd.concat(answers)
    df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/kialoAnswers/merged_answers.csv",
              index=False, sep="\t")


def get_nodeID_partial_textmatch(id2info, partialtext):
    for id, info in id2info.items():
        text = info["text"]
        if partialtext in text:
            return id
    for id, info in id2info.items():
        text = info["text"]
        initial_part = partialtext.split(",")[0]
        if initial_part in text:
            return id
    return None


def get_annotations_with_id(answer_dir, max_documents, child2info, cand2info, input_file):
    child2candidates = get_child2candidates(input_file)
    print(child2info)
    annot1 = 0
    annot2 = 1
    annot3 = 2
    dic = {"child": [], "candidate": [], "annotation1": [], "annotation2": [], "annotation3": [], "childID": [],
           "candidateID": [], "goldParent": []}
    for i in range(1, max_documents + 1):
        answersheet = pd.read_csv("%s/KialoAnnotations%d.csv" % (answer_dir, i), sep=",")
        for item in answersheet.columns[1:]:
            child = None
            candidate = None
            for poss_child, poss_cand in child2candidates.items():
                if poss_child in item:
                    child = poss_child
                    for _, candi in poss_cand.items():
                        if candi in item:
                            candidate = candi
            child_id = get_nodeID_partial_textmatch(child2info, child)
            candidate_id = get_nodeID_partial_textmatch(cand2info, candidate)
            dic["child"].append(child)
            dic["candidate"].append(candidate)
            dic["annotation1"].append(answersheet[item].values[annot1])
            dic["annotation2"].append(answersheet[item].values[annot2])
            dic["annotation3"].append(answersheet[item].values[annot3])
            dic["childID"].append(child_id)
            dic["candidateID"].append(candidate_id)
            dic["goldParent"].append(child2info[child_id]["parentID"])
    df = pd.DataFrame().from_dict(dic)
    df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.csv", index=False,
              sep="\t")


def get_child2candidates(input_file):
    df_input = pd.read_excel(input_file)
    child2candidates = {}
    for i in range(len(df_input)):
        child = df_input["form questions"].values[i]
        candidates = df_input["answer start"].values[i].split("#")
        child2candidates[child] = {}
        for j in range(len(candidates)):
            child2candidates[child][j] = candidates[j]
    return child2candidates


def get_number_of_candidates_per_doc(answer_dir, max_documents):
    index2candidates = {}
    for i in range(1, max_documents + 1):
        answersheet = pd.read_csv("%s/KialoAnnotations%d.csv" % (answer_dir, i), sep=",")
        index2candidates[i] = len(answersheet.columns) - 1
    return index2candidates


def convert_stringlabels_to_int(labels):
    labeldic = {'BEST PARENT': 1, 'SUITABLE PARENT': 2, 'LESS SUITABLE PARENT': 3}
    return [labeldic[el] for el in labels]


def get_weighted_kappa(weights, annot1, annot2):
    weight_matrix = np.zeros(shape=[3, 3])

    weight_matrix[0][1] = weights[0]
    weight_matrix[0][2] = weights[2]
    weight_matrix[1][0] = weights[0]
    weight_matrix[1][2] = weights[1]
    weight_matrix[2][0] = weights[2]
    weight_matrix[2][1] = weights[1]
    return kappa(y_true=annot1, y_pred=annot2, weights=weight_matrix)


def filter_annotation_frame(frame, type=None, depth_bounday=None):
    child2info, cand2info = get_annotation_info_dic()
    child_type = []
    child_depth = []
    for i in range(len(frame)):
        id = frame.childID.values[i]
        if id in child2info:
            info = child2info[id]
            node_type = info['child_type']
            node_level = info['level']
            child_type.append(node_type)
            child_depth.append(node_level)
        else:
            child_depth.append(None)
            child_type.append(None)
    frame["type"] = child_type
    frame["depth"] = child_depth
    frame.dropna(inplace=True)
    if type:
        new_frame = frame[frame.type == type]
    elif depth_bounday:
        if depth_bounday == "small":
            new_frame = frame[frame.depth <= 4]
        else:
            new_frame = frame[frame.depth > 4]
    return new_frame


def compute_performance(input_file, type=None, depth=None):
    df = pd.read_csv(input_file, sep="\t")
    # df.dropna(inplace=True)
    if type or depth:
        df = filter_annotation_frame(df, type, depth)
    df = filter_annotation_frame(df, None, None, average_distance=True)
    annotators = ["annotation1", "annotation2", "annotation3"]
    ranks = {"annotation1": [], "annotation2": [], "annotation3": []}
    for i in range(len(df)):
        candidateID = df.candidateID.values[i]
        parentID = df.goldParent.values[i]

        if candidateID == parentID:
            for annot in annotators:
                if df[annot].values[i] == "BEST PARENT":
                    ranks[annot].append(1)
                elif df[annot].values[i] == "SUITABLE PARENT":
                    ranks[annot].append(2)
                else:
                    ranks[annot].append(1000)
    for annotator in annotators:
        annotations = ranks[annotator]
        prec1 = eval.precision_at_rank(annotations, 1)
        prec5 = eval.precision_at_rank(annotations, 5)
        print("annotator: %s; precision at rank1: %.2f; precision at rank5: %.2f" % (annotator, prec1, prec5))

    # for annotator in annotators:


def compute_agreement(input_file, type=None, depth=None):
    """

    :param input_file: a csv file with each row having a child parent pair and the corresponding annotation for each annotator
    :return:
    """
    df = pd.read_csv(input_file, sep="\t")
    if type or depth:
        df = filter_annotation_frame(df, type, depth)

    # df.dropna(inplace=True)
    annotators = ["annotation1", "annotation2", "annotation3"]
    # iterate through the annotators and construct agreement matrix
    # for now: for each pair drop the rows were one of the annotators has no annotation
    # for kappa: try different values for the weights
    agreement_matrices = {"weighted kappa": np.zeros(shape=[3, 3]), "spearman correlation": np.zeros(shape=[3, 3]),
                          "F1Macro": np.zeros(shape=[3, 3]), "F1BestParent": np.zeros(shape=[3, 3]),
                          "F1SuitableParent": np.zeros(shape=[3, 3]), "F1LessSuitable": np.zeros(shape=[3, 3]),
                          "PercentageAgreementBestParent": np.zeros(shape=[3, 3])}
    for i in range(len(annotators)):
        for j in range(len(annotators)):
            if i != j:
                # filter the annotation frame and drop nana
                annot_pair_df = df[[annotators[i], annotators[j]]]
                annot_pair_df.dropna(inplace=True)
                annotations1 = list(annot_pair_df[annotators[i]].values)
                annotations2 = list(annot_pair_df[annotators[j]].values)
                annotations_as_int1 = convert_stringlabels_to_int(annotations1)
                annotations_as_int2 = convert_stringlabels_to_int(annotations2)
                corr = correlation(y_true=annotations_as_int1, y_pred=annotations_as_int2, corr_type="spearman")
                f1_macro = f1_score(y_true=annotations1, y_pred=annotations2, average='macro')
                class_based_f1 = classification_report(y_true=annotations1, y_pred=annotations2, output_dict=True)
                best_parent = class_based_f1["BEST PARENT"]["f1-score"]
                less_suitable = class_based_f1["LESS SUITABLE PARENT"]["f1-score"]
                suitable = class_based_f1["SUITABLE PARENT"]["f1-score"]

                weighted_kappa = get_weighted_kappa(weights=[0.2, 0.3, 0.5], annot1=annotations_as_int1,
                                                    annot2=annotations_as_int2)
                percentage_best_parent_agreement = 0
                for index1 in range(len(annotations_as_int1)):
                    if annotations_as_int1[index1] == 1:
                        if annotations_as_int2[index1] == 1:
                            percentage_best_parent_agreement += 1
                percentage_best_parent_agreement = percentage_best_parent_agreement / annotations_as_int1.count(1)
                agreement_matrices["F1Macro"][i][j] = f1_macro
                agreement_matrices["spearman correlation"][i][j] = corr
                agreement_matrices["weighted kappa"][i][j] = weighted_kappa
                agreement_matrices["F1SuitableParent"][i][j] = suitable
                agreement_matrices["F1BestParent"][i][j] = best_parent
                agreement_matrices["F1LessSuitable"][i][j] = less_suitable
                agreement_matrices["PercentageAgreementBestParent"][i][j] = percentage_best_parent_agreement
    return agreement_matrices


def plot_agreement_matrices():
    results = compute_agreement(
        input_file="/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv",
        depth="small")
    sns.set(font_scale=1.5)
    labels = ["ann1", "ann2", "ann3"]
    for metric, matrix in results.items():
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues").set_title("%s" % metric)
        plt.tight_layout()
        plt.savefig("/Users/falkne/PycharmProjects/deliberatorium/data/plots/annotations_study/large/%s.png" % metric,
                    dpi=800)
        # plt.show()
        plt.clf()


def get_annotation_info_dic():
    dir = "/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/annotation_kialoV2"
    child2info = {}
    candidate2info = {}
    for d in os.listdir(dir):
        topic = d
        for subdir in os.listdir("%s/%s" % (dir, d)):
            example_path = "%s/%s/%s" % (dir, d, subdir)
            candidates_info = pd.read_csv("%s/candidates.csv" % example_path, sep="\t")
            map_id = int(candidates_info["id"].values[0])
            gold = pd.read_csv("%s/target_node.csv" % example_path, sep="\t")
            map = KialoMap("/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english/%s.pkl" % map_id)
            child_id = gold["ID"].values[0]
            parent_id = gold["parentID"].values[0]
            childnode = [el for el in map.all_children if el.id == str(child_id)]
            child_text = gold["target node"].values[0]

            old_id = None
            if not childnode:
                for el in map.all_children:
                    if child_text in el.name:
                        old_id = child_id
                        child_id = el.id
                        childnode = [el for el in map.all_children if el.id == str(child_id)]
            childtype = childnode[0].type
            child2info[child_id] = {"topic": topic, "map_id": map_id, "child_type": childtype, "child_id": child_id,
                                    "map_size": map.number_of_children(), "map_depth": map.max_depth,
                                    "parentID": parent_id, "wrong_id": old_id, "text": child_text,
                                    "level": childnode[0].get_level()}
            for i in range(len(candidates_info)):
                text = candidates_info.nodes[i]
                # text = text.replace("[", "").replace("]", "")
                level = candidates_info.levels[i]
                nodetype = candidates_info.node_type[i]
                currentid = candidates_info.id[i]
                distance = candidates_info.distance[i]
                impact = candidates_info.impact[i]
                candidate2info[currentid] = {"level": level, "candidate_type": nodetype, "candidate_id": currentid,
                                             "shortest_distance": distance, "impact": impact, "text": text}
    return child2info, candidate2info


def get_performance_for_every_setup():
    compute_performance("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv")
    print("\t")
    compute_performance("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv", type=1)
    print("\t")
    compute_performance("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv",
                        type=-1)
    print("\t")

    compute_performance("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv",
                        depth="small")
    print("\t")

    compute_performance("/Users/falkne/PycharmProjects/deliberatorium/annotation/pilot/anntotation_with_id.tsv",
                        depth="large")


def get_used_maps_ids(input_file):
    df = pd.read_csv(input_file, sep="\t")
    ids = df.childID.values
    map_ids = set([int(el) for el in ids])
    return map_ids


def create_annotation_batch():
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


def create_plots(map_df, target_df, candidate_df):
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
    import matplotlib.pyplot as plt
    import seaborn as sns

    convert_annotationdata_to_googleforms(
        "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation200instances")

    # 200
    # add topics: concrete
    #

    # 8 per map (4 general, 4 specific)
    # keep 5 topics
    # confidence rating
    # keep track of annotator
