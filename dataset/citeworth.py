
import json


#                         'paper_id': metadata['paper_id'],
#                         'section_title': sec['section'],
#                         'paper_title' : metadata['title'],
#                         'paper_abstract' : metadata['abstract'],
#                         'paper_year' : metadata['year'],
#                         'outgoing_citations' : metadata['outbound_citations'],
#                         'outgoing_citations_in_section' : cits_in_section


def rename_key(data, old : str, new :str) -> None:
    for entry in data:
        entry[new] = entry.pop(old)


def load_citeworth(path):
    """ Loads a single file """
    print("Loading citworth data from", path)
    data = list()
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            data.append(json.loads(l.strip()))

    rename_key(data,'paper_title','title')
    rename_key(data,'paper_authors','authors')
    rename_key(data, 'paper_year','year')
    rename_key(data, 'outgoing_citations', 'references')
    rename_key(data,'paper_id','id')

    for entry in data:
        if entry['year'] and entry['year'] != None:
            entry['year'] = int(entry['year'])

    # get all ids
    all_ids = []
    for entry in data:
        all_ids.append(entry['id'])

    # if ref id not in dataset remove
    remove_cnt = 0
    for entry in data:
        for ref in entry['references']:
            if ref not in all_ids:
                entry['references'].remove(ref)
                remove_cnt += 1

    print(f"Removed {remove_cnt} orphaned references")

    return data
