
import json

SYNONYM_DICT = {
    "introduction": "introduction",
    "intro": "introduction",
    "overview": "introduction",
    "motivation": "introduction",
    "problem motivation": "introduction",
    "related work": "related work",
    "previous work": "related work",
    "literature": "related work",
    "background": "related work",
    "literature review": "related work",
    "state of the art": "related work",
    "current state of research": "related work",
    "requirement": "related work",
    "experiment": "experiment",
    "experimental result": "experiment",
    "experimental setup": "experiment",
    "result": "experiment",
    "result and analysis": "experiment",
    "evaluation": "experiment",
    "performance evaluation": "experiment",
    "experiment and result": "experiment",
    "analysis": "experiment",
    "methodology": "method",
    "method": "method",
    "material and method": "method",
    "proposed method": "method",
    "evaluation methodology": "method",
    "procedure": "method",
    "implementation": "method",
    "experimental design": "method",
    "implementation detail": "method",
    "system model": "method",
    "discussion": "discussion",
    "limitation": "discussion",
    "result and discussion": "discussion",
    "future work": "conclusion",
    "conclusion": "conclusion",
    "summary": "conclusion",
    "discussion and conclusion": "conclusion",
    "conclusion and outlook": "conclusion",
    "conclusion and future work": "conclusion",
    "concluding remark": "conclusion",
    "empirical result": "result",
    "experiment result": "experiment",
    "conclusion limitation and future work": "conclusion",
    "conclusion and future direction": "conclusion",
    "relation to prior work": "related work",
    "background and related work": "related work",
    "experimental evaluation": "experiment",
    "technical background": "related work",
    "model": "method",
    "result and evaluation": "experiment",
    "discussion and future work": "discussion",
    "related work and background": "related work",
    "future work and conclusion": "conclusion",
    "conclusion and discussion": "conclusion",
    "introduction and motivation": "introduction",
    "evaluation and result": "experiment",
    "proposed approach": "method",
    "related work and conclusion": "conclusion",
    "conclusion and limitation": "conclusion",
    "motivation and related work": "related work",
    "introduction and related work": "introduction",
    "summary and conclusion": "conclusion",
    "related literature": "related work",
    "conclusion and perspective": "conclusion",
    "introduction and background": "introduction",
    "proposed methodology": "method",
    "review of previous method": "related work",
    "discussion and outlook": "discussion"
}

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


def load_citeworth(path, use_synonym_dict = True, section_divided = True):
    """ Loads a single file """
    print("Loading citworth data from", path)
    data = list()
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            data.append(json.loads(l.strip()))

    rename_key(data,'paper_title','title')
    # rename_key(data,'paper_authors','authors')
    rename_key(data, 'paper_year','year')
    if section_divided:
        rename_key(data, 'outgoing_citations_in_paragraph', 'references')
    else:
        rename_key(data, 'outgoing_citations', 'references')

    rename_key(data,'paper_id','id')

    # apply synonym dict
    if use_synonym_dict:
        for entry in data:
            entry['section_title'] = SYNONYM_DICT[entry['section_title'].lower()]

    for entry in data:
        if entry['year'] and entry['year'] != None:
            entry['year'] = int(entry['year'])

    # get all ids
    all_ids = []
    for entry in data:
        all_ids.append(entry['id'])



    return data
