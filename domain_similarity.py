import pandas as pd
import seaborn as sns
from argumentMap import KialoMap, DeliberatoriumMap
import os
import itertools
import operator
from pathlib import Path
import spacy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# define the POS tags of content words (vs function words)
from kialo_domains_util import get_maps2uniquetopic, get_map2topics

CONTENTWORD_POSTAGS = {"noun", "verb", "adj", "propn", "adv"}
IT_MAPS = {"Doparia sulla legge elettorale (m1)", "Doparia sulla legge elettorale (m2)", "Naples Biofuels Discussion"}
main_topics = {"religion", "gender", "enviroment", "animals", "democracy", "europe", "technology", "education",
               "history", "violence", "war", "business", "human rights", "health", "morality", "philosophy",
               "science"}


def read_vocab_from_map(map, only_content_words=False):
    """
    This method reads in all text from a map and creates a vocabulary object
    :param map: an ArgumentMap object
    :param only_content_words: if set to true the lemma list will only contain content words (adjectives, nouns..)
    :return: a list of lemmata extracted from the content of a map
    """
    all_text = [node.name for node in map.all_children]
    # use the medium model trained on web data from spacy, use english (translations have to be used for e.g. IT maps)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmas = []
    for doc in nlp.pipe(all_text, batch_size=20):
        lemma_list = [str(tok.lemma_).lower() for tok in doc]
        pos_tags = [str(tok.pos_).lower() for tok in doc]
        if only_content_words:
            lemma_list = [lemma_list[i] for i in range(len(pos_tags)) if pos_tags[i] in CONTENTWORD_POSTAGS]
        lemmas.append(lemma_list)
    lemmas = list(itertools.chain(*lemmas))
    return lemmas


def build_vectorizer(map2lemmas):
    """
    Create a vectorizer based on word frequency of a corpus. Use the content of all given maps and merge all docuemnts
    (aka maps) into one big corpus. Create the vectorizer based on that.
    :param map2lemmas: a dictionary containing the maps with a list of their lemmas to build the vectorizer on
    :return: a tf-idf vectorizer object, minimum frequency is 3
    """
    corpora = []
    for _, lemma_list in map2lemmas.items():
        as_string = " ".join(lemma_list)
        corpora.append(as_string)
    vectorizer = TfidfVectorizer(min_df=3)
    vectorizer.fit(corpora)
    return vectorizer


def get_map_lemmas(maps, translated_map_path=None):
    """Read all deliberatorium or kialo maps and return a dictionary
     containing each map-name and a list of corresponding lemmata"""
    maps2lemmas = {}
    for _, map in maps.items():
        if map.name in IT_MAPS and translated_map_path:
            lemmas = read_vocab_from_file(
                translated_map_path % map.name, True)
        else:
            lemmas = read_vocab_from_map(map, True)
        maps2lemmas[map.name] = lemmas
    return maps2lemmas


def read_vocab_from_file(path, only_content_words=False):
    """
    Read files and extract a list of lemmata
    :param path: the path to read the file from
    :param only_content_words: if only content words should be considered (adjectives, nouns...)
    :return: a list of lemmata
    """
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmas = []
    all_text = open(path).readlines()
    all_text = [line.strip() for line in all_text]
    for doc in nlp.pipe(all_text, batch_size=20):
        lemma_list = [str(tok.lemma_).lower() for tok in doc]
        pos_tags = [str(tok.pos_).lower() for tok in doc]
        if only_content_words:
            lemma_list = [lemma_list[i] for i in range(len(pos_tags)) if pos_tags[i] in CONTENTWORD_POSTAGS]
        lemmas.append(lemma_list)
    lemmas = list(itertools.chain(*lemmas))
    return lemmas


def get_tfidf_vector(lemmas, vectorizer):
    """
    Extract the tf-idf vector of a document
    :param lemmas: a list of lemmata representing a document
    :param vectorizer: a vectorizer used to transform the document
    :return: the tf-idf vector of the given document
    """
    corpus = " ".join(lemmas)
    vector = vectorizer.transform([corpus]).toarray()
    return vector


def get_top_ranked_words(vector, n, vectorizer):
    """
    Extract words with the highest tf-idf score given a tf-idf vector of a document / map.
    :param vector: the tf-idf document vector of a corpus
    :param n: the number of most significant words to be returned
    :param vectorizer: the vectorizer object which contains the vocab index
    :return: the most significant (in terms of tf-idf) n words of the document /map
    """
    feature_array = np.array(vectorizer.get_feature_names())
    tfidf_sorting = np.argsort(vector.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]
    return top_n


def shared_vocab(lemmas_1, lemmas_2):
    """Return the set of shared vocabulary between two lists of lemmas"""
    shared_vocab = set(lemmas_1).intersection(set(lemmas_2))
    return shared_vocab


def tfidf_similairty(vector_1, vector_2):
    """Return the cosine similarity between two vectors"""
    similarity = cosine_similarity(vector_1, vector_2)
    return similarity


def read_all_maps():
    """read all maps from deliberatorium and kialo, return them as two dictionaries"""
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/kialo_maps"
    maps = os.listdir(data_path)
    kialo_maps = {}
    for i in tqdm(range(len(maps))):
        map = maps[i]
        path = "%s/%s" % (data_path, map)
        argument_map = KialoMap(path)
        kialo_maps[argument_map.name.strip()] = argument_map

    data_path = Path.home() / "data/e-delib/deliberatorium/maps/deliberatorium_maps"
    maps = os.listdir(data_path)

    deliberatorium_maps = {}
    for map in maps:
        argument_map = DeliberatoriumMap("%s/%s" % (str(data_path), map))
        deliberatorium_maps[argument_map.name] = argument_map
    return kialo_maps, deliberatorium_maps


def tag_headmap_kialo(path_kialo2topics, n, outpath, cut_most_frequent=False):
    """Heatmap for the top 50 most frequent topics"""
    topics = pd.read_csv(path_kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topics["topic_tags"] = [[w.strip().lower() for w in el.split(",")] for el in topics.topics.values]
    top50 = dict(Counter(itertools.chain(*topics.topic_tags.values)).most_common(n))
    vocab = list(top50.keys())
    if cut_most_frequent:
        vocab = vocab[10:]
    print("vocab size is %d" % len(vocab))
    vocab_matrix = np.zeros(shape=[len(vocab), len(vocab)])
    vocab2index = dict(zip(vocab, range(len(vocab))))
    for map_tags in topics.topic_tags.values:
        for tag1 in map_tags:
            if tag1 in vocab2index:
                index1 = vocab2index[tag1]
            for tag2 in map_tags:
                if tag2 in vocab2index:
                    index2 = vocab2index[tag2]
                    if index1 != index2:
                        vocab_matrix[index1][index2] += 1

    sns.set(font_scale=0.5)
    ax = sns.heatmap(vocab_matrix, xticklabels=vocab, yticklabels=vocab)
    plt.tight_layout()
    plt.savefig(outpath, dpi=1500)


def main_domains(path_kialo2topics, maintopics):
    map2unique, _ = get_maps2uniquetopic(path_kialo2topics, maintopics)
    word_frequency = Counter(map2unique.values()).most_common()
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]
    plt.barh(words, counts)
    plt.title("distribution of unique domains")
    plt.ylabel("number of maps")
    plt.xlabel("Domains")
    plt.tight_layout()
    plt.savefig("data/plots/distribution_domains2.png", dpi=1500)

    # heat_map(vocab=list(set(sub2maintopic.values())), map2topics=maps2maintopic,
    #         out_path="data/plots/main_topics_heatmap.png")


def kialo_maps2topic(path_kialo2topics, maintopics):
    map2unique, _ = get_maps2uniquetopic(path_kialo2topics, maintopics)
    kialomaps, _ = read_all_maps()
    topic2numberofnodes = defaultdict(int)
    smallest = 1000
    for mapname, topic in map2unique.items():
        if mapname not in kialomaps:
            print(mapname)
        else:
            map = kialomaps[mapname]
            nodes = map.number_of_children()
            if map.number_of_children() < smallest:
                smallest = map.number_of_children()
                smalles_map = mapname
            topic2numberofnodes[topic] += nodes
    print(smalles_map)
    print(smallest)
    print(topic2numberofnodes)
    topic2numberofnodes = {k: v for k, v in sorted(topic2numberofnodes.items(), key=lambda item: item[1])}
    plt.barh(list(topic2numberofnodes.keys()), list(topic2numberofnodes.values()))
    plt.title("number of nodes per domain")
    plt.ylabel("domain")
    plt.xlabel("number of nodes")
    plt.tight_layout()
    plt.savefig("data/plots/node_distribution_domains.png", dpi=1500)


def heat_map(vocab, map2topics, out_path):
    """
    Create a heat map of co-ocurrences of topics
    :param vocab: the topics to be considered
    :param map_topics: a dictionary with a map name and the corresponding topics
    :param out_path: the path to save the heatmap to
    """
    vocab_matrix, _ = get_cooccurrence_matrix(vocab, map2topics)
    sns.set(font_scale=0.5)
    sns.heatmap(vocab_matrix, xticklabels=vocab, yticklabels=vocab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1500)


def get_cooccurrence_matrix(vocab, map2topics):
    """
    Retrieve a co-ocurrence matrix of joint frequency of topic tags
    :param vocab: the vocabulary to consider for the matrix
    :param map2topics: all maps with their corresponding topics
    :return: the co-occ. matrix and the word2index
    """
    vocab_matrix = np.zeros(shape=[len(vocab), len(vocab)])
    vocab2index = dict(zip(vocab, range(len(vocab))))
    for map_name, map_topics in map2topics.items():
        for tag1 in map_topics:
            if tag1 in vocab2index:
                index1 = vocab2index[tag1]
            for tag2 in map_topics:
                if tag2 in vocab2index:
                    index2 = vocab2index[tag2]
                    if index1 != index2:
                        vocab_matrix[index1][index2] += 1
    return vocab_matrix, vocab2index


def create_domains(path_kialo2topics, outpath):
    map2topics = get_map2topics(path_kialo2topics)
    top50 = dict(Counter(itertools.chain(*map2topics.values())).most_common(50))
    vocab = list(top50.keys())
    vocab = vocab[10:]
    vocab_matrix, vocab2index = get_cooccurrence_matrix(vocab, map2topics)
    index2vocab = dict(zip(range(len(vocab)), vocab))
    wordjointwords = defaultdict(set)
    # collect co-ocurrences of <= 15
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if i != j:
                joint_freq = vocab_matrix[i][j]
                if joint_freq >= 15:
                    word1 = index2vocab[i]
                    word2 = index2vocab[j]
                    wordjointwords[word1].add(word2)
                    wordjointwords[word2].add(word2)
                    wordjointwords[word2].add(word1)
                    wordjointwords[word1].add(word1)
    word2replacement = {}
    # create labels that consist of the tags that co-occur more than 15 times
    for k, v in wordjointwords.items():
        for k2, v2 in wordjointwords.items():
            if k != k2:
                subsest = v.intersection(v2)
                if len(subsest) >= 1:
                    v.update(v2)
                    word2replacement[k] = "/".join(sorted(list(v)))
    replaced_vocab = list(set([word2replacement[w] if w in word2replacement else w for w in vocab]))
    print("reduced vocab is %d" % len(replaced_vocab))
    heat_map(vocab=replaced_vocab, map2topics=map2topics, out_path=outpath)


def most_frequent_topics(path_kialo2topics, n):
    """Plot the most frequent topic tags with their frequency"""
    maps2topics = get_map2topics(path_kialo2topics)
    all_topics = itertools.chain(*maps2topics.values())
    word_frequency = Counter(all_topics).most_common(n)
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]
    plt.barh(words, counts)
    plt.title("%d most frequent topic tags in kialo" % n)
    plt.ylabel("Frequency")
    plt.xlabel("Words")
    plt.show()


def plot_cosine_similarities(similarities):
    """Plot a histogram of cosine similarities"""
    plt.hist(similarities, density=True, bins=30)  # density=False would make counts
    plt.ylabel('density')
    plt.xlabel('cosine similarity')
    plt.savefig('document_similarity_histogram.png')


def compute_document_similarity(source_maps, target_maps, vectorizer):
    """
    Compute 2 similarity dictionaries between each map in the source maps and each map in the target maps:
    one based on the number of shared vocabulary and one based on the tf-idf document similarity
    :param source_maps: a dictionary with a source map name and the corresponding lemma list
    :param target_maps: a dictionary with a target map name and a corresponding lemma list
    :param vectorizer: a tf-dif vectorizer
    :return: a dictionary with every source map and the shared vocab to every target map, a dictionary with every source map and the tf-idf sim to every target map
    """
    shared_vocab_dic = defaultdict(dict)
    tfidf_dic = defaultdict(dict)
    for source_map_name, source_map_lemmas in source_maps.items():
        pbar = tqdm(total=len(target_maps))
        source_tfidf_vec = get_tfidf_vector(source_map_lemmas, vectorizer)
        for target_map_name, target_map_lemmas in target_maps.items():
            pbar.update(1)
            shared_vocab_dic[source_map_name][target_map_name] = len(shared_vocab(source_map_lemmas, target_map_lemmas))
            target_tfidf_vec = get_tfidf_vector(target_map_lemmas, vectorizer)
            similarity = np.round(tfidf_similairty(source_tfidf_vec, target_tfidf_vec), decimals=2)[0][0]
            tfidf_dic[source_map_name][target_map_name] = similarity
    return shared_vocab_dic, tfidf_dic


def get_most_similar_documents(sim_dic, n):
    """for a given map return the most similar documents (based on the given similarity dictionary)"""
    return sorted(sim_dic.items(), key=operator.itemgetter(1), reverse=True)[:n]


def get_topic_similarity_heatmapTFIDF_top50():
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialomaps2maintopics.tsv"
    topics = pd.read_csv(kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topics["topic_tags"] = [[w.strip().lower() for w in el.split(",")] for el in topics.topics.values]
    top50 = dict(Counter(itertools.chain(*topics.topic_tags.values)).most_common(50))
    vocab = list(set(list(top50.keys())))
    map2topic = get_map2topics(kialo2topics)
    domain2similarities = get_cosine_similarity_matrix(map2topic, top50=vocab)
    plot_heatmap_tfidf_similarity(domain2similarities, vocab,
                                  "/Users/falkne/PycharmProjects/deliberatorium/data/plots/tfIDF_similarity_top50outdomain.png")


def get_cosine_similarity_matrix(map2topics, top50=None):
    tfidf_vectors = np.load("/Users/falkne/PycharmProjects/deliberatorium/kialo_tfidf_vectorsminfreq3.npy")
    tfidf_vectors = np.squeeze(tfidf_vectors, axis=1)
    # create the vocab matrix based on all maps
    maps = open("data/map_names.txt", "r").readlines()
    maps = [el.strip() for el in maps]
    map2id = dict(zip(maps, range(0, len(maps))))
    # get a full matrix of pair-wise cosine similarities between all maps (based on their tf-idf vectors)
    map2map_similarity_matrix = cosine_similarity(tfidf_vectors, tfidf_vectors, dense_output=True)
    # create a dic that will keep track of each topic and all similarities of that topic to other topics
    domain2similarities = defaultdict(dict)

    # iterate through all topics and append the similarity between each map and each other map, keeping track of the topic
    for m, topic1 in map2topics.items():
        if m in map2id:
            id1 = map2id[m]
            for m2, topic2 in map2topics.items():
                if m2 in map2id and m2 != m:
                    id2 = map2id[m2]
                    cosine_sim = map2map_similarity_matrix[id1][id2]
                    if top50:
                        for single_topic1 in topic1:
                            for single_topic2 in topic2:
                                if single_topic1 in set(top50) and single_topic2 in set(top50):
                                    if topic2 not in domain2similarities[topic1]:
                                        domain2similarities[topic1][topic2] = []
                                    domain2similarities[topic1][topic2].append(cosine_sim)
                    else:
                        if topic2 not in domain2similarities[topic1]:
                            domain2similarities[topic1][topic2] = []
                        domain2similarities[topic1][topic2].append(cosine_sim)
    return domain2similarities


def plot_heatmap_tfidf_similarity(topics2similarities, vocab, save_path):
    """
    Creates a Triangle Heat Map for a list if topics (dictionary of pair-wise domain similarities has to be provided)
    :param topics2similarities: a nested dictionary, each key is a topic (topic1) and each value is a dictionary with each domain as a key again (topic2)
    and a list of cosine similarities that have been computed based on pair-wise similarities between all maps of topic1 and all maps of topic2
    :param vocab: a unique list of "domains" to create the heatmap for
    :param save_path: a path to store the plot to
    """
    vocab2index = dict(zip(vocab, range(len(vocab))))
    vocab_matrix = np.zeros(shape=[len(vocab), len(vocab)])

    for tag, index in vocab2index.items():
        for tag2, index2 in vocab2index.items():
            sims = topics2similarities[tag][tag2]
            average = np.average(sims)
            vocab_matrix[index][index2] = average

    matrix = np.triu(vocab_matrix)
    sns.set(font_scale=0.3)
    sns.heatmap(vocab_matrix, xticklabels=vocab, yticklabels=vocab, center=0,
                square=False, linewidths=.5, cbar_kws={"shrink": .8}, annot=False, mask=matrix)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1500)


def createTextBasedDocumentsFromMaps():
    """create txt files that contain the text from every argument map from kialo"""
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/kialo_maps"
    maps = os.listdir(data_path)
    dir = "/Users/falkne/PycharmProjects/deliberatorium/kialo_docs/"
    for map in maps:
        path = "%s/%s" % (data_path, map)
        outpath = open("%s/%s" % (dir, map), "w")
        argument_map = KialoMap(path)
        thread = argument_map.name
        outpath.write(thread)
        for node in argument_map.all_children:
            text = node.name
            if not "-> " in text:
                outpath.write(text + "\n")
        outpath.close()


def create_heatmap_tfIDFsimilarity_mainDomains():
    # create a mapping from Argument map to main topic
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialomaps2maintopics.tsv"
    main_topics = "/Users/falkne/PycharmProjects/deliberatorium/data/kialo_domains.tsv"
    map2unique, topic2submapping = get_maps2uniquetopic(kialo2topics, main_topics)
    # a list of unique main topics
    main_tops = list(set(list(map2unique.values())))
    # load the TF-IDF vectors for all Kialo maps

    # iterate through all topics and append the similarity between each map and each other map, keeping track of the topic
    domain2similarities = get_cosine_similarity_matrix(map2unique)
    plot_heatmap_tfidf_similarity(domain2similarities, main_tops,
                                  "/Users/falkne/PycharmProjects/deliberatorium/data/plots/tfIDF_similarity_maintopic.png")

# vocab overlap between each map within deliberatorium
# vocab overlap for each map in delib to the kialo maps
# most important tf-idf content words for each delib map, number of unique content words
