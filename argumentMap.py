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
        self._all_children = self.init_children()
        self._events = None

    def load_json(self, json_file):
        """Loads the json object from the json file"""
        try:
            with open(json_file, encoding='utf-8') as f:
                json_obj = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(json_file, encoding='utf-8-sig') as f:
                json_obj = json.load(f)
        return json_obj

    def init_children(self):
        """Retrieves all nodes from the argument map. Creates a list based on the retrieved node list that contains the nodes as Child Objects"""
        all_children = []
        children_list = []
        if self._jsonObj["children"]:
            all_children = ArgumentMap.get_children([], self._jsonObj["children"])
        for c in all_children:
            children_list.append(ChildNode(c))
        return children_list

    def number_of_children(self):
        """Returns the number of child nodes in the map"""
        return len(self._all_children)

    @staticmethod
    def get_children(all_children, node):
        """Recursive method: recursively travers through a node and its' children.
        Append all children to a list as a tuple together with the corresponding parent node.
        List grows incrementally"""
        if type(node) == dict:
            if not "children" in node:
                return all_children
            elif node["children"] == None:
                return all_children
            else:
                for child in node["children"]:
                    all_children.append(child)
                    ArgumentMap.get_children(all_children, child)
            # if the child is a list of children do recursion for every child
        else:
            for child in node:
                all_children.append(child)
                ArgumentMap.get_children(all_children, child)
        return all_children

    def __str__(self):
        return str(self._name)
