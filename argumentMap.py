import json
import re
from abc import ABC, abstractmethod
from childNode import DelibChildNode, KialoChildNode
from encode_nodes import MapEncoder


class ArgumentMap(ABC):
    """
    Abstract class for Argument Maps. Specific ArgumentMaps can inherit from this base class. The base class
    defines the core of every argument map, which is a method to load the data from source, children, a name and an ID.
    An ArgumentMap is a directed Graph with node objects. Ever node has exactly one parent (except for the root node) and
    can have zero to several children.
    """

    def __init__(self, data_path, label=None):
        """
        Initialize the Map given the path to the data to load from. Every map has the following attributes:
        id: a unique identifier, e.g. 'E-1R8DN19-3171'
        name: a name, e.g. 'Scholio DC2: New Zealand Massacre Highlights Global Reach of White Extremism'
        Args:
            data_path: path to the file that contains the data of the argument map.
        """
        self.label = label
        self._data = self.load_data(data_path)
        self._id = self._data["id"]
        self._name = self._data["name"]
        self._direct_children = self.init_children()
        all_children = []
        # create a list that stores all nodes of a map by iterating through the first level of nodes
        # and calling the recursive method for each child.
        for child in self._direct_children:
            all_children = self.get_all_children(node=child, child_list=all_children)
        self._all_children = all_children

    @abstractmethod
    def load_data(self, data_path) -> dict:
        """This method should read data from a file and return a dictionary object storing all details needed
        to construct a map."""

    @abstractmethod
    def init_children(self) -> list:
        """Method to initialize the Nodes that are directly attached to the root. Should return a list of ChildNodes"""

    def get_all_children(self, node, child_list):
        """recursively iterate trough all nodes of a map and append the children and their children..."""
        child_list.append(node)
        if node._is_leaf:
            return child_list
        else:
            for childnode in node._direct_children:
                self.get_all_children(childnode, child_list)
        return child_list

    def number_of_children(self):
        """Returns the number of child nodes in the map"""
        return len(self._all_children)

    def __str__(self):
        return str(self._name)


class DeliberatoriumMap(ArgumentMap):
    """
    An ArgumentMap for Maps from Deliberatorium
    """

    def __init__(self, data_path, label=None):
        """
        Additional attributes for this type of Map are:
        description: a more detailed description, can be 'None'
        :param data_path: the path to the json file of the argument map
        """
        super(DeliberatoriumMap, self).__init__(data_path, label)
        self._description = self._data["description"]

    def load_data(self, json_file):
        """Loads the json object from the json file"""
        try:
            with open(json_file, encoding='utf-8') as f:
                json_obj = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(json_file, encoding='utf-8-sig') as f:
                json_obj = json.load(f)
        return json_obj

    def init_children(self):
        """Initializes the first level of children = all nodes that are directly located at the root of the map"""
        children_list = []
        if self._data["children"]:
            for child in self._data["children"]:
                children_list.append(DelibChildNode(child))
        return children_list


class KialoMap(ArgumentMap):

    def __init__(self, data_path, label=None):
        super(KialoMap, self).__init__(data_path, label)
        self._max_depth = self.get_max_depth()

    def load_data(self, data_path):
        """Loads the data from the .txt files"""
        data_dict = {}
        with open(data_path, 'r') as f:
            content = f.readlines()
        # the name is the topic of the map
        data_dict["name"] = content[2].replace("1. ", "")
        try:
            # the id can be taken from the file name
            map_id = re.search("\d+", data_path).group(0)
        except AttributeError:
            map_id = 0  # default id if no id found
        data_dict["id"] = map_id
        data_dict["children"] = []
        # skip first two lines (line 1 contains the discussion topic and line 2 is empty)
        for line in content[3:]:
            # each line contains the content of one child node.
            parts = line.split(" ")
            if len(parts) > 1:
                id = line.split(" ")[0]
                type = line.split(" ")[1].replace(":", "").strip()
                text = " ".join(line.split(" ")[2:]).strip()
                data_dict["children"].append({"id": id, "type": type, "name": text})
        # for each child the list of direct children has to be added to the node dictionary (which can only
        # be retrieved after having read the file completely.
        for child in data_dict["children"]:
            child["children"] = self.get_direct_children(child["id"], data_dict["children"])
        return data_dict

    def init_children(self):
        """Initializes the first level of children = all nodes that are directly located at the root of the map"""
        children_list = []
        if self._data["children"]:
            for child in self._data["children"]:
                if child["id"].count(".") == 2:
                    children_list.append(KialoChildNode(child))
        return children_list

    def get_direct_children(self, id, all_nodes):
        """Given an id, extract all child nodes that correspond to direct children of a node
        (e.g. if my id is '1' then give me any two digits with 1 as a first """
        # each dot represents a node level, so the direct children are one level above the current node (node level +1)
        child_len = id.count(".") + 1
        # get all nodes that are at exactly one level above and that are children of the node
        # (the id starts with the parents node id)
        direct_children = [node for node in all_nodes if
                           node["id"].count(".") == child_len and node["id"].startswith(id)]
        return direct_children

    def get_max_depth(self):
        """Return the maximal tree depth of this map"""
        return max([node._depth for node in self._all_children])
