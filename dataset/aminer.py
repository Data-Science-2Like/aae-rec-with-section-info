import json


def load_dblp(path):
    """ Loads a single file """
    print("Loading dblp data from", path)
    with open(path, 'r') as fhandle:
        obj = [json.loads(line.rstrip('\n')) for line in fhandle]
    return obj


def load_acm(path):
    """ Loads a single file """
    print("Loading acm data from", path)
    with open(path, 'r', encoding="utf8") as fhandle:
        obj = []
        paper = {}
        paper["references"] = []

        for line in fhandle:
            line = line.rstrip('\n')

            if len(line) == 0:
                obj.append(paper)
                paper = {}
                paper["references"] = []

            elif line[1] == '*':
                paper["title"] = line[2:]
            elif line[1] == '@':
                paper["authors"] = line[2:].split(",")
            elif line[1] == 't':
                paper["year"] = int(line[2:])
            elif line[1] == 'c':
                paper["venue"] = line[2:]
            elif line[1] == 'i':
                paper["id"] = line[6:]
            else:
                paper["references"].append(line[2:])

    return obj