from event import Event
from abc import ABC, abstractmethod


class ChildNode(ABC):
    """
    The abstract class ChildNode. A ChildNode has an ID, a name (which corresponds to a textual representation of the
    content of the node) and a type (e.g. Pro). Each node has information about whether it has children or
    whether it is a leaf node.
    """

    def __init__(self, node_dict):
        """
        Represent the class ChildNode. A ChildNode has the following attributes:
        id: a unique identifier, e.g. 'E-1Q7SF3H-1623' or '1.12'
        name: a name, e.g. 'What would an impartial observer, with access to data from social media, find out about the sources of radicalism and hate speech?'
        type: what kind of node, can be issue, idea, pro, con
        is_leaf: boolean stores whether the node is a leaf
        direct_children: if the node has children; a list of ChildNode objects that are directly attached to this node.
        parent: exactly one node that is the parent of this node (or None if it is the root node)
        embedding: either None if it has not been encoded yet, or an embedding representation of this node
        """
        self._id = node_dict["id"]
        self._name = node_dict["name"]
        self._type = node_dict["type"]
        self._direct_children = self.init_direct_children(node_dict)
        # if child list is empty this node is a leaf node
        self._is_leaf = True if len(self._direct_children) == 0 else False
        self._parent = None
        self._embedding = None
        # set this node as parent for each of its' children
        for child in self._direct_children:
            child._parent = self

    def add_embedding(self, embedding):
        """adds an embedding representation for this node, sets flag to true"""
        self._embedding = embedding

    def __str__(self):
        return str(self._name)

    @abstractmethod
    def init_direct_children(self, node_dict) -> list:
        """"""


class DelibChildNode(ChildNode):

    def __init__(self, node_dict):
        """
        DelibChildNode as specific additional attributes that are only present in this kind of Node.
        description: more detailed content of the node.
        events: a list of Events, each associated with its own attributes (see OBJ Event)
        :param node_dict: the json representation of the node
        """
        super(DelibChildNode, self).__init__(node_dict)

        self._description = node_dict["description"]
        self._creator = node_dict["creator"]
        self._events = self.init_events(node_dict["events"])

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes, returns the empty list if no children"""
        child_list = []
        if "children" in node_dict and node_dict["children"] != None:
            children = node_dict["children"]
            for c in children:
                child_list.append(DelibChildNode(c))
        return child_list

    def init_events(self, event_node):
        """Initialize all events that are connected to this child"""
        event_nodes = []
        for event in event_node:
            event_nodes.append(Event(event))
        return event_nodes

    def number_events(self):
        """Returns the number of events connected to this child"""
        return len(self._events)


class KialoChildNode(ChildNode):
    """
    The Kialo argument map is a simple deliberation map. The nodes can either be of type 'Pro' or type 'Con'
    Each node can have further children. Each node is associated with a tree depth.
    """

    def __init__(self, node_dict):
        """
        The KialoChildNode has one additional attribute
        depth: The tree depth of this node
        :param node_dict: the dictionary representation of the node
        """
        super(KialoChildNode, self).__init__(node_dict)
        self._depth = self.get_depth()

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes, returns the empty list if no children"""
        child_list = []
        if "children" in node_dict and node_dict["children"] != None:
            children = node_dict["children"]
            for c in children:
                child_list.append(KialoChildNode(c))
        return child_list

    def get_depth(self):
        """Returns the depth of this node"""
        return self._id.count(".")
