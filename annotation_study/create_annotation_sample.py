import itertools

import pandas as pd
import os
import random
from utils import read_all_maps, filter_maps_by_main_topic, filter_maps_by_depth, filter_maps_by_size
from util import remove_url_and_hashtags
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
    output_file = open(annot_dir + "/googleforms_backup.tsv", "w")
    output_file.write(
        "form questions\tquestion type\tanswer start\tanswer end\tDescription\tclaimID\ttargetID\tcandidateIDs\n")
    confidence_row = "Confidence Rating\tSCALE\t1,3,Unsure,Certain\t\tHow confident are you in your response?\t\t\t\n"
    for root, dirs, files in os.walk(annot_dir):
        for file in files:
            if "example" in file:
                example_file = os.path.join(root, file)
                map_info = pd.read_csv(example_file.replace(file, "map_info.txt"), sep="\t", dtype=str)
                claim = map_info["claim"].values[0]
                mapID = map_info["mapID"].values[0]
                df = pd.read_csv(example_file, sep="\t", dtype=str)
                target = df["target"].values[0]
                candidates = list(df["candidates"].values)
                # remove any additional line breaks in candidates and target and claim
                claim = claim.replace("\n", " ")
                target = target.replace("\n", " ")
                target = target.replace("#", " ")
                candidates = [el.replace("\n", " ") for el in candidates]
                candidates = [el.replace("#", " ") for el in candidates]
                targetID = df["targetID"].values[0]
                candidateIDs = list(df["candidateID"].values)
                canidate_string = "#".join(candidates)
                id_string = "##".join([str(el) for el in candidateIDs])
                selectstring = "BEST PARENT#SUITABLE PARENT#LESS SUITABLE PARENT"
                output_file.write("%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n" % (
                    target, "checkbox grid", canidate_string, selectstring, claim, mapID, targetID, id_string))
                output_file.write(confidence_row)

    output_file.close()


def convert_backup_to_googleforms():
    # form questions	question type	answer start	answer end	Description
    output_file = open("/Users/falkne/PycharmProjects/deliberatorium/data/annotationBackup/googleforms_backup.tsv", "w")
    output_file.write(
        "form questions\tquestion type\tanswer start\tanswer end\tDescription\tmapID\ttargetID\tcandidateIDs\tparentID\ttargetType\n")
    confidence_row = "Confidence Rating\tSCALE\t1,3,Unsure,Certain\t\tHow confident are you in your response?\t\t\n"
    motivation_row = "Motivation\tshort answer\t\tWhat led you to choose this as the best parent?\t\t\t\n"
    selectstring = "BEST PARENT#SUITABLE PARENT#LESS SUITABLE PARENT"

    counter = 0
    for root, dirs, files in os.walk("/Users/falkne/PycharmProjects/deliberatorium/data/annotationBackup"):
        for file in files:
            if counter == 10:
                break
            if "example" in file:
                example_file = os.path.join(root, file)
                map_info = pd.read_csv(example_file.replace(file, "map_info.txt"), sep="\t", dtype=str)
                target_info = pd.read_csv(example_file.replace(file, "target_node.csv"), sep="\t", dtype=str)
                target_type = target_info["target type"].values[0]
                parent_id = target_info["parentID"].values[0]
                claim = map_info["claim"].values[0]
                mapID = map_info["mapID"].values[0]
                df = pd.read_csv(example_file, sep="\t", dtype=str)
                target = df["target"].values[0]
                candidates = list(df["candidates"].values)
                # remove any additional line breaks in candidates and target and claim
                claim = claim.replace("\n", " ")
                target = target.replace("\n", " ")
                target = target.replace("#", " ")
                candidates = [el.replace("\n", " ") for el in candidates]
                candidates = [el.replace("#", " ") for el in candidates]
                targetID = df["targetID"].values[0]
                candidateIDs = list(df["candidateID"].values)
                canidate_string = "#".join(candidates)
                id_string = "##".join([str(el) for el in candidateIDs])
                output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                    target, "checkbox grid", canidate_string, selectstring, claim, mapID, targetID, id_string,
                    target_type, parent_id))
                output_file.write(confidence_row)
                output_file.write(motivation_row)
                counter += 1

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


def create_backup_batch():
    topics = ["gender", "economics", "immigration", "politics", "environment"]
    used_maps = set(pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_maps.csv",
                                sep="\t").mapID.values)
    outpath = "/Users/falkne/PycharmProjects/deliberatorium/data/annotationBackup"
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"
    argument_maps = read_all_maps(data_path)
    argument_maps = [map for map in argument_maps if map.id not in used_maps]
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/dataKialoV2/mapID2topicTags/ENmapID2topicTags.txt"
    main_topics = "/Users/falkne/PycharmProjects/deliberatorium/dataKialoV2/maintopic2subtopic.tsv"
    map2unique, topic2submapping = get_maps2uniquetopic(kialo2topics, main_topics)
    for topic in topics:
        sample_annotation_batch_tolerant(topic=topic, argument_maps=argument_maps,
                                         output_dir="%s/%s" % (outpath, topic),
                                         number_small_maps=2,
                                         number_larger_maps=1,
                                         map2unique=map2unique)


def final_batch():
    df = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances__tmp.tsv", sep="\t",
        dtype=str)
    argument_maps = read_all_maps(data_path)
    affected_ids = set([int(el.split(".")[0]) for el in df.corrected_childiD.values])
    print(affected_ids)
    affected_maps = [map for map in argument_maps if map.id in affected_ids]
    id2map = {}
    for map in affected_maps:
        id2map[map.id] = map
    print(id2map)
    childids = []
    candidateids = []
    childtexts = []
    candidatetext = []
    parentids = []
    descriptions = []
    for index, row in df.iterrows():
        childID = row.corrected_childiD
        candidateIDs = row.corrected_candidate_nodes
        parentID = row.parentID
        candidate_node_ids = candidateIDs.split("##")
        descriptions.append(row.Description)
        map = id2map[int(childID.split(".")[0])]
        candidate_nodes = [el for el in map.all_children if
                           el.id in candidate_node_ids]

        child_node = [el for el in map.all_children if
                      str(el.id) == str(childID)][0]
        candidate_texts = [remove_url_and_hashtags(el.name) for el in candidate_nodes]
        canidate_string = "#".join(candidate_texts)
        child_text = remove_url_and_hashtags(child_node.name)
        childids.append(childID)
        candidateids.append(candidateIDs)
        childtexts.append(child_text)
        candidatetext.append(canidate_string)
        parentids.append(parentID)
    d = pd.DataFrame()
    d["childID"] = childids
    d["candidateIDs"] = candidateids
    d["form questions"] = childtexts
    d["answer start"] = candidatetext
    d["Description"] = descriptions
    d["parentID"] = parentids
    d.to_csv("annotation_100_instances.tsv", sep="\t", index=False)
    # output_file.write("%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n" % (
    #    target, "checkbox grid", canidate_string, selectstring, claim, mapID, targetID, id_string))


def final_google_forms():
    df = pd.read_csv("annotation_100_instances.tsv", sep="\t", dtype=str)
    outfile = open("kialo_100_instances_googleform.csv", "w")
    outfile.write(
        "form questions\tquestion type\tanswer start\tanswer end\tDescription\tchildID\tcandidateIDs\n")
    confidence_row = "Confidence Rating\tSCALE\t1,3,Unsure,Certain\t\tHow confident are you in your response?\t\t\n"
    motivation_row = "Motivation\tshort answer\t\tWhat led you to choose this as the best parent?\t\t\t\n"
    selectstring = "BEST PARENT#SUITABLE PARENT#LESS SUITABLE PARENT"
    for i in range(len(df)):
        outfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            df["form questions"].values[i], "checkbox grid", df["answer start"].values[i], selectstring,
            df["Description"].values[i], df["childID"].values[i], df["candidateIDs"].values[i]))
        outfile.write(motivation_row)
        outfile.write(confidence_row)
    outfile.close()


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


# extract all files with the data for the annotation
# filter out by the IDs that are affected
# read the corresponding map and get a random node
# replace the affected node with the new node

def fix_duplicates():
    annot_frame = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances2.tsv", sep="\t")
    all_candidates = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_candidates.csv", sep="\t")
    all_targets = pd.read_csv(

        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv", sep="\t")
    affected_sameID = [208, 228, 250, 290, 304]
    affected_samecandidate = [306, 296, 268, 266, 206]
    argument_maps = read_all_maps(data_path, debug_maps_size=1100)

    affected_ids = [annot_frame.claimID.values[el] for el in affected_samecandidate]
    print(affected_ids)
    affected_map_ids = [int(el) for el in affected_ids]
    print(affected_map_ids)
    affected_maps = [map for map in argument_maps if map.id in affected_map_ids]
    id2map = {}
    for map in affected_maps:
        id2map[map.id] = map
    affected = annot_frame[annot_frame.claimID.isin(affected_ids)]
    # annot_frame = annot_frame.dropna()
    affected = affected.dropna()
    print(affected.columns)
    # print(all_candidates.columns)
    from collections import Counter
    for i in range(len(affected)):
        # childID = affected_ids[i]
        childID = affected.claimID.values[i]
        currentMap = id2map[int(childID)]

        candidates = affected.candidateIDs.values[i]

        candidates = candidates.split("##")
        duplicate = Counter(candidates).most_common()[0]
        # candidates = [float(el) for el in candidates]

        candidate_nodes = [el for el in currentMap.all_children if
                           str(el.id) in candidates]

        child_node = [el for el in currentMap.all_children if
                      str(el.id) == str(childID)][0]
        # print(candidates)
        print("unique candidates")
        print(len(set(candidates)))
        distances = [child_node.shortest_path(el) for el in candidate_nodes]
        print(distances)
        close_candidates = [candidate for candidate in currentMap.all_children if
                            child_node.shortest_path(candidate) <= 3 and child_node.shortest_path(
                                candidate) > 1 and str(candidate.id) not in candidates]
        if len(close_candidates) > 0:
            node = random.sample(close_candidates, 1)[0]
        else:
            node = random.sample([el for el in currentMap.all_children if
                                  child_node.shortest_path(el) > 1 and str(el.id) not in candidates], 1)[0]
        print("map")
        print(currentMap.id)
        print(currentMap.name)

        print("child ID")
        print(childID)
        print(remove_url_and_hashtags(child_node.name))
        print("original candidates")
        print(candidates)
        print("duplicate")
        print(duplicate)

        print("new candidate")
        print(node.id)
        print(remove_url_and_hashtags(node.name))
        print(node.shortest_path(child_node))


def get_new_sample():
    from argumentMap import KialoMap

    map = KialoMap("/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english/33292.pkl")
    for node in map.all_children:
        # 29992.312  ##29992.8##29992.631##29992.363##29992.359##29992.94##29992.346##29992.310##29992.310##29992.361
        # 33292.76  ##33292.170##33292.49##33292.170##33292.70##33292.172##33292.78##33292.35##33292.184##33292.47
        if node.id == "33292.186":
            s = random.sample(map.all_children, 1)[0]
            print(s)
            print(s.id)
            print(s.name)
            print(s.shortest_path(node))


def check_annotion_sample(file):
    from pathlib import Path
    from argumentMap import KialoMap
    df = pd.read_csv(file, sep="\t", dtype=str)
    map_ids = df["claimID"].values
    map_ids = set([el.split(".")[0] for el in map_ids])
    print(map_ids)
    # extract the maps that are used in the study
    maps = list(Path(data_path).glob(f"*.pkl"))
    used_maps = {}
    for map_file in maps:
        map_id = str(map_file).split("/")[-1].replace(".pkl", "")
        if map_id in map_ids:
            used_maps[map_id] = KialoMap(str(str(map_file)))
    df = df.dropna()
    # create a ID2node dic
    id2node = {}
    candidate_ids = list([cand.split("##") for cand in df.corrected_cand_IDs.values])
    candidatetexts = list([cand.split("#") for cand in df["answer start"].values])
    child_texts = df["form questions"].values
    for i in range(len(df)):
        childID = df.corrected_child_ID.values[i]
        map = used_maps[childID.split(".")[0]]

        if childID not in id2node:
            child_node = [el for el in map.all_children if
                          el.id == childID]
            id2node[childID] = child_node[0]

        sub_candidate_ids = candidate_ids[i]
        for candidateID in sub_candidate_ids:
            if candidateID not in id2node:
                node = [el for el in map.all_children if
                        el.id == candidateID]
                if not node:
                    print(childID)
                    print(candidateID)
                    print("CAND not found")

                id2node[str(candidateID)] = node[0]

    # iterate over annotation and extract the nodes and all info
    number_unique_candidates = []
    parent_in_candidates = []
    childID_in_candidates = []
    candidate_texts_are_equal = []
    child_text_is_equal = []
    corrected_childiD = []
    corrected_candidate_nodes = []
    parent_ids = []
    print(df.columns)
    for index, row in df.iterrows():
        # ID in the file
        claimID = str(row.corrected_child_ID)
        # ID of the candidates
        candidates = row.corrected_cand_IDs
        candidate_ids = candidates.split("##")

        number_unique_candidates.append(len(set(candidate_ids)))
        # the texts in the file
        candidate_texts = row["answer start"].split("#")
        # the nodes to the
        candidate_nodes = [id2node[el] for el in candidate_ids]
        print(len(candidate_nodes))
        corrected_cand_ids = [str(el.id) for el in candidate_nodes]
        corrected_candidate_nodes.append("##".join(corrected_cand_ids))
        candidate_texts_equal = True
        for i in range(len(candidate_texts)):
            text1 = candidate_texts[i]
            candnode = candidate_nodes[i]
            if remove_url_and_hashtags(candnode.name) != text1:
                candidate_texts_equal = False
        candidate_texts_are_equal.append(candidate_texts_equal)
        child_node = id2node[claimID]
        corrected_childiD.append(str(child_node.id))
        if remove_url_and_hashtags(child_node.name) != row["form questions"]:
            child_text_is_equal.append(False)
        else:
            child_text_is_equal.append(True)
        if claimID in candidate_ids:
            childID_in_candidates.append(True)
        else:
            childID_in_candidates.append(False)
        distances = [el.shortest_path(child_node) for el in candidate_nodes]
        if child_node.parent:
            parent_ids.append(str(child_node.parent.id))
        else:
            # print(child_node)
            # print(child_node.id)
            parent_ids.append("None")
        if 1 in distances:
            parent_in_candidates.append(True)
        else:
            parent_in_candidates.append(False)
    df["number_unique_candidates"] = number_unique_candidates
    df["parent_in_candidates"] = parent_in_candidates
    df["childID_in_candidates"] = childID_in_candidates
    df["candidate_texts_are_equal"] = candidate_texts_are_equal
    df["child_text_is_equal"] = child_text_is_equal
    df["corrected_childiD"] = corrected_childiD
    df["corrected_candidate_nodes"] = corrected_candidate_nodes
    df["parentID"] = parent_ids

    df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_check.tsv",
              sep="\t", index=False)


def correct_ids():
    df = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv",
        sep="\t", dtype=str)
    df = df.dropna()
    print(len(df))
    candidates = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_candidates.csv", sep="\t",
        dtype=str)
    targets = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
                          sep="\t", dtype=str)

    corrected_candidate_nodes = []
    corrected_child_ids = []
    print(targets.columns)
    updated_cand_ids = {"33292.70", "28423.29", "25000.634", "29992.310", "28996.123", "33292.110", "28996.213",
                        "28996.115", "29992.627", "25000.376"}
    print(list(candidates.ID.values))
    for index, row in df.iterrows():
        candidate_texts = row["answer start"].split("#")
        child_text = row["form questions"]
        candidate_ids = row.candidateIDs.split("##")
        wrong_id = row.claimID
        child_match = targets[targets.ID == wrong_id]
        if len(child_match) == 1:
            corrected_child_ids.append(wrong_id)
        else:
            wrong_id = wrong_id + "0"
            child_match = targets[targets.ID == wrong_id]
            if len(child_match) == 1:
                corrected_child_ids.append(wrong_id)
        new_candidate_ids = []
        for cand_id in candidate_ids:
            found = False

            if cand_id in updated_cand_ids:
                new_candidate_ids.append(cand_id)
                found = True
            else:
                cand_match = candidates[(candidates.ID == cand_id) & (candidates.childID == wrong_id)]
                if len(cand_match) >= 1 and len(set(cand_match.ID)) == 1:
                    new_candidate_ids.append(cand_id)
                    found = True
                else:
                    new_id = "%s0" % cand_id
                    cand_match = candidates[(candidates.ID == new_id) & (candidates.childID == wrong_id)]
                    if len(cand_match) >= 1 and len(set(cand_match.ID)) == 1:
                        new_candidate_ids.append(new_id)
                        found = True
                    else:
                        new_id = "%s0" % new_id
                        cand_match = candidates[(candidates.ID == new_id) & (candidates.childID == wrong_id)]
                        if len(cand_match) >= 1 and len(set(cand_match.ID)) == 1:
                            new_candidate_ids.append(new_id)
                            found = True
                        else:
                            new_id = "%s0" % new_id
                            cand_match = candidates[(candidates.ID == new_id) & (candidates.childID == wrong_id)]
                            if len(cand_match) >= 1 and len(set(cand_match.ID)) == 1:
                                new_candidate_ids.append(new_id)
                                found = True
        new_candidate_ids = "##".join(new_candidate_ids)
        # print(new_candidate_ids)
        corrected_candidate_nodes.append(new_candidate_ids)

    df["corrected_child_ID"] = corrected_child_ids
    df["corrected_cand_IDs"] = corrected_candidate_nodes
    df.to_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv",
        sep="\t", index=False)

    # print(len(matching_node))


if __name__ == '__main__':
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"

    topic2claims = "/Users/johannesfalk/PycharmProjects/deliberatorium/data/annotation_kialoV2"
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoID2MainTopic.csv"
    main_topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialo_domains.tsv"
    convert_backup_to_googleforms()
    # analyze_annotation_sample()
    # fix_duplicates()
    # final_google_forms()
    # correct_ids()
    # check_annotion_sample(
    #    "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv")

    # things to check:
    # the unique length of the candidates is 10
    # the child is not in the candidates
    # the ids of the candidates are correct and the corresponding text is correct
    # the ids of the children are correct and the corresponding text
    # the parent is amongst the candidates
