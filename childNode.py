from event import Event


class ChildNode:

    def __init__(self, node_dict):
        """
        Represent the class ChildNode. A ChildNode has the following attributes:
        id: a unique identifier, e.g. 'E-1Q7SF3H-1623'
        name: a name, e.g. 'What would an impartial observer, with access to data from social media, find out about the sources of radicalism and hate speech?'
        type: what kind of node, can be issue, idea, pro, con
        description: more detailed content of the node.
        :param node_dict: the json representation of the node
        :param parent: the json representation of the parent node
        """
        self._id = node_dict["id"]
        self._name = node_dict["name"]
        self._description = node_dict["description"]
        self._type = node_dict["type"]
        self._creator = node_dict["creator"]
        self._events = self.init_events(node_dict["events"])
        self._direct_children = self.init_direct_children(node_dict)
        self._has_embedding = False

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes"""
        children = node_dict["children"]
        child_list = []
        if children:
            for c in children:
                child_list.append(ChildNode(c))
            return child_list
        return None

    def init_events(self, event_node):
        """Initialize all events that are connected to this child"""
        event_nodes = []
        for event in event_node:
            event_nodes.append(Event(event))
        return event_nodes

    def number_events(self):
        """Returns the number of events connected to this child"""
        return len(self._events)

    def is_leaf(self):
        """Returns True if the child is a leaf, has no children, False otherwise"""
        if not self._direct_children:
            return True
        return False

    def add_embedding(self, embedding):
        """adds an embedding representation for this node, sets flag to true"""
        self._embedding = embedding
        self._has_embedding = True

    def __str__(self):
        return str(self._name)
