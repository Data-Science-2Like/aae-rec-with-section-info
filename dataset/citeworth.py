
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
    rename_key(data, 'paper_year','year')
    rename_key(data, 'outgoing_citations', 'references')
    rename_key(data,'paper_id','id')

    for entry in data:
        if entry['year'] and entry['year'] != None:
            entry['year'] = int(entry['year'])
    return data
