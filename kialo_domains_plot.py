from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from kialo_domains_util import get_maps2uniquetopic


def plot_mapsPerTopic(path_kialo2topics, maintopics, outpath):
    map2unique, _ = get_maps2uniquetopic(path_kialo2topics, maintopics)
    d = Counter(map2unique.values())
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    keys = list(d.keys())
    # get values in the same order as keys, and parse percentage values
    vals = list(d.values())
    g = sns.barplot(x=keys, y=vals)
    g.set_xticklabels(g.get_xticklabels(), rotation=30, fontsize=7, horizontalalignment='right')
    plt.savefig(outpath, dpi=1000)


if __name__ == '__main__':
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kilaoV2Langs/ENmapID2topicTags.txt"
    maintopics = "/Users/falkne/PycharmProjects/deliberatorium/data/kilaoV2Langs/maintopic2subtopic.tsv"
    outpath = "/Users/falkne/PycharmProjects/deliberatorium/data/kilaoV2Langs/topics/topicdistribution.png"
    mapt2unique, (map2topics, _, _) = get_maps2uniquetopic(kialo2topics, maintopics)
    # for map, tag in mapt2unique.items():
    #    if tag == "politics":
    #        othertags = map2topics[map]
    #        print(othertags)
    plot_mapsPerTopic(kialo2topics, maintopics, outpath)
