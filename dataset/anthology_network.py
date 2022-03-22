def load_aan(path):
    """ Loads a single file """
    print("Loading aan data from", path)
    obj = {}

    with open(path / "paper_ids.txt", 'r', encoding="utf8") as f:
        for line in f:
            paper = {}
            paper["references"] = []
            line = line.rstrip('\n')

            parts = line.split('\t')

            if len(parts) < 3: # this file is horribly formatted
                continue
            # id is first entry, last is published year, the rest is the title
            paper["id"] = parts[0]
            paper["year"] = int(parts[-1])
            paper["title"] = parts[1]

            obj[parts[0]] = paper

    citation_count = 0
    with open(path / "networks" / "paper-citation-network.txt", 'r', encoding="utf8") as f:
        for line in f:
            line = line.rstrip('\n')
            parts = line.split(' ')
            citation_count += 1
            assert len(parts) == 3  # first entry is citing paper , third is cited paper

            # maybe it didn't get parsed right
            if parts[0] in obj.keys():
                obj[parts[0]]['references'].append(parts[2])

    print(f"Loaded {citation_count} citation edges")

    return obj.values()
