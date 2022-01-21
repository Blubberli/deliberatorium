from sentence_transformers import SentenceTransformer
import torch
import pickle


class MapEncoder:
    """
    This class contains the functionality to create sentence embeddings from a pretrained SBERT model.
    The sentence embeddings can be computed and the embedding representation for each node can be added to a
    specific argument map.
    """

    def __init__(self, sbert_model_identifier, max_seq_len, use_descriptions=False, normalize_embeddings=False,
                 model=None):
        """

        :param sbert_model_identifier: a pretrained sbert model, e.g. all-MiniLM-L6-v2
        :param max_seq_len: the maximum length to be encoded (the default in the model was 128)
        :param use_descriptions: some nodes have additional information as a 'description'. if set to True these will be
        added to the sentence representation of a node, default is False
        :param normalize_embeddings: whether to normalize the sentence embeddings to unit length. If true, dot product
        can be used to compute similarity. Default is false.
        """
        self._use_descriptions = use_descriptions
        self._device = self.get_device()
        self._sbertModel = model if model else SentenceTransformer(sbert_model_identifier, device=self._device)
        self._max_len = max_seq_len
        self._sbertModel.max_seq_length = self._max_len
        self._normalize_embeddings = normalize_embeddings

        print("loaded sbert model from %s" % sbert_model_identifier)

    def get_device(self):
        """:returns cuda or cpu as  a string to specify the device on which the embeddings are computed"""
        if torch.cuda.is_available():
            device = "cuda"
            print('GPU in use:')
        else:
            print('using the CPU')
            device = "cpu"
        return device

    def encode_argument_map(self, argument_map):
        """
        loads an argument map from a path. adds the sbert embedding representation to each node.
        :param path: [str] the location of the argument map to be encoded
        :return: a dictionary with embddings, corresponding sentences and corresponding IDs
        """
        nodes = argument_map._all_children
        sentences = [node._name for node in nodes]
        unique_ids = [node._id for node in nodes]
        if self._use_descriptions:
            descriptions = [node._description for node in nodes]
            sentences = [
                sentences[i] + " : " + descriptions[i] if (descriptions[i] != '' and descriptions[i] != None) else
                sentences[i] for i in
                range(len(sentences))]
        embeddings = self._sbertModel.encode(sentences, show_progress_bar=True,
                                             normalize_embeddings=self._normalize_embeddings)
        for i in range(len(nodes)):
            nodes[i].add_embedding(embeddings[i])
        return {"embeddings": embeddings, "sentences": sentences, "ID": unique_ids}

    def add_stored_embeddings(self, argument_map, path_to_pckl):
        """Given a path of pregenerated embeddings, add"""
        data = self.load_embeddings(path_to_pckl=path_to_pckl)
        nodes = argument_map._all_children
        for node in nodes:
            id = node._id
            embedding = data[id]
            node.add_embedding(embedding)

    @staticmethod
    def save_embeddings(path_to_pckl, embeddings, unique_id):
        """store dictionary with embeddings, ids and sentences to a pickle object"""
        with open(path_to_pckl, "wb") as fOut:
            pickle.dump(dict(zip(unique_id, embeddings)), fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_embeddings(path_to_pckl):
        """load dictionary with sentences, embeddings and IDs from a pickle object"""
        with open(path_to_pckl, "rb") as fIn:
            stored_data = pickle.load(fIn)
            # stored_sentences = stored_data['sentences']
            # stored_embeddings = stored_data['embeddings']
            # stored_ids = stored_data["ID"]
        return stored_data
        # return {"sentences": stored_sentences, "embeddings": stored_embeddings, "ID": stored_ids, "ID2EMBEDDINGS":}
