import os
from argumentMap import ArgumentMap


def read_all_maps(path_to_maps):
    """Read all maps from a directory and store them in a dictionary with ID as key and map as value"""
    maps = os.listdir(path_to_maps)
    id2map = {}
    for map in maps:
        argument_map = ArgumentMap("%s/%s" % (path_to_maps, map))
        id2map[argument_map._id] = argument_map
    return id2map


def get_id2map_name(id2map):
    """Get dictionary with ID as key and map name as value"""
    id2map_name = {}
    for id, map in id2map.items():
        id2map_name[id] = map._name
    return id2map_name


def get_possible_event_actions(id2map):
    """Iterate through all maps to track all possible event actions. Return them as a set"""
    actions = []
    for id, map in id2map.items():
        children = map._all_children
        for child in children:
            events = child._events
            for e in events:
                actions.append(e._action)
    return set(actions)


def get_attributes_of_map(map):
    """Return all attributes tracked with a map"""
    attributes = []
    for child in map._all_children:
        events = child._events
        for e in events:
            attributes.append(e._attributes)
    return attributes


def get_all_issues(map):
    """Return all issues of a map"""
    issues = []
    for child in map._all_children:
        if child._type == "issue":
            issues.append(child)
    return issues


def get_ideas(map):
    """Return all ideas of a map"""
    ideas = []
    for child in map._all_children:
        if child._type == "idea":
            ideas.append(child)
    return ideas
