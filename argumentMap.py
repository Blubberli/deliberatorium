import json
from childNode import ChildNode


class ArgumentMap:

    def __init__(self, json_file):
        """
        represents the class 'ArgumentMap'. An argument map has
        jsonObj: a json object that contains all content of the argument map
        id: a unique identifier, e.g. 'E-1R8DN19-3171'
        name: a name, e.g. 'Scholio DC2: New Zealand Massacre Highlights Global Reach of White Extremism'
        description: a more detailed description, can be 'None'
        type: should be 'map'
        children: a number of nodes, called 'children'
        :param json_file: the path to the json file of the argument map
        """
        self._jsonObj = self.load_json(json_file)
        self._id = self._jsonObj["id"]
        self._name = self._jsonObj["name"]
        self._description = self._jsonObj["description"]
        self._type = self._jsonObj["type"]
        self._direct_children = self.init_children()
        all_children = []
        # create a list that stores all nodes of a map by iterating through the first level of nodes and calling the recursive method for each child.
        for child in self._direct_children:
            all_children = self.get_all_children(node=child, child_list=all_children)
        self._all_children = all_children

    def load_json(self, json_file):
        """Loads the json object from the json file"""
        try:
            with open(json_file, encoding='utf-8') as f:
                json_obj = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(json_file, encoding='utf-8-sig') as f:
                json_obj = json.load(f)
        return json_obj

    def get_all_children(self, node, child_list):
        """recursively iterate trough all nodes of a map and append the children and their children..."""
        child_list.append(node)
        if node._is_leaf:
            return child_list
        else:
            for childnode in node._direct_children:
                self.get_all_children(childnode, child_list)
        return child_list

    def init_children(self):
        """Initializes the first level of children = all nodes that are directly located at the root of the map"""
        children_list = []
        if self._jsonObj["children"]:
            for child in self._jsonObj["children"]:
                children_list.append(ChildNode(child))
        return children_list

    def number_of_children(self):
        """Returns the number of child nodes in the map"""
        return len(self._all_children)

    def __str__(self):
        return str(self._name)
