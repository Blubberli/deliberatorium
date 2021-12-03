from event import Event


class ChildNode:

    def __init__(self, node_dict):
        """
        Represent the class ChildNode. A ChildNode has the following attributes:
        id: a unique identifier, e.g. 'E-1Q7SF3H-1623'
        name: a name, e.g. 'What would an impartial observer, with access to data from social media, find out about the sources of radicalism and hate speech?'
        type: what kind of node, can be issue, idea, pro, con
        description: more detailed content of the node.
        events: a list of Events, each associated with its own attributes (see OBJ Event)
        direct_children: empty list if no children or a list of Child Nodes
        is_leaf: True if the node has no children, otherwise False
        parent: either None if the node is attached to the root, or one Node that is the direct parent of this node
        embedding: either None if it has not been encoded yet, or an embedding representation of this node
        :param node_dict: the json representation of the node
        """
        self._id = node_dict["id"]
        self._name = node_dict["name"]
        self._description = node_dict["description"]
        self._type = node_dict["type"]
        self._creator = node_dict["creator"]
        self._events = self.init_events(node_dict["events"])
        self._direct_children = self.init_direct_children(node_dict)
        # if child list is empty this node is a leaf node
        self._is_leaf = True if len(self._direct_children) == 0 else False
        self._parent = None
        self._embedding = None
        # set this node as parent for each of its' children
        for child in self._direct_children:
            child._parent = self

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes, returns the empty list if no children"""
        child_list = []
        if "children" in node_dict and node_dict["children"] != None:
            children = node_dict["children"]
            for c in children:
                child_list.append(ChildNode(c))
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

    def add_embedding(self, embedding):
        """adds an embedding representation for this node, sets flag to true"""
        self._embedding = embedding

    def __str__(self):
        return str(self._name)
