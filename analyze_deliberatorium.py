import os
from argumentMap import ArgumentMap
from collections import Counter


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


def get_overall_action_freqs(id2map):
    """Returns a frequency dictionary with each action and their overall frequency"""
    actions = []
    for id, map in id2map.items():
        children = map._all_children
        for child in children:
            events = child._events
            for e in events:
                actions.append(e._action)
    return Counter(actions)


def get_specific_action_freq_of_map(map, action):
    """Given a map and an action (e.g. 'EDITED') this method returns the frequency of the action in the given map"""
    map_actions = []
    children = map._all_children
    for child in children:
        events = child._events
        for e in events:
            map_actions.append(e._action)
    if action in Counter(map_actions):
        return Counter(map_actions)[action]
    else:
        print("action not in this map")
        return None


def get_events_of_map(map):
    """Return all events tracked with a map"""
    events_list = []
    for child in map._all_children:
        events = child._events
        for e in events:
            events_list.append(e)
    return events_list


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


def get_pros_and_cons(map):
    """returns all pro and con arguments of a map"""
    pros, cons = [], []
    for child in map._all_children:
        if child._type == "pro":
            pros.append(child)
        elif child._type == "con":
            cons.append(child)
    return pros, cons


if __name__ == '__main__':
    id2map = read_all_maps("argument_maps")
    file = open("data/actions.csv", "w")
    file.write("ACTION\tFREQ\n")
    for id, map in id2map.items():
        children = map._all_children
        issues = get_all_issues(map)

        idea = get_ideas(map)

        events = get_events_of_map(map)

        pros, cons = get_pros_and_cons(map)
        print("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n" % (
            id, map._name, map.number_of_children(), len(issues), len(idea), len(pros), len(cons), len(events)))
    f = get_overall_action_freqs(id2map)
    for action, freq in f.items():
        file.write(action + "\t" + str(freq) + "\n")
    file.close()
