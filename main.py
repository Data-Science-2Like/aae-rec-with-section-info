"""
Executable to run AAE on the AMiner DBLP dataset
"""
import argparse
import glob
import itertools
import json
import os
from pathlib import Path
from typing import Optional
from random import shuffle

from gensim.models.keyedvectors import KeyedVectors
from joblib import Parallel, delayed

import models.aae

from models.aae import AAERecommender, DecodingRecommender
from models.baselines import Countbased, RandomBaseline, MostPopular, BM25Baseline
# from models.baselines import BM25Baseline, RandomBaseline
from models.datasets import Bags
from models.evaluation import Evaluation
from models.svd import SVDRecommender
from models.vae import VAERecommender
from models.dae import DAERecommender
from models.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition

from utils.paths import W2V_PATH, W2V_IS_BINARY, ACM_PATH, CITEWORTH_PATH, DBLP_PATH, CITE2_PATH, CITE5_PATH, AAN_PATH, CITE5_PAPERS_PATH, CITE7_PATH
from dataset.aminer import load_dblp, load_acm
from dataset.citeworth import load_citeworth
from dataset.anthology_network import load_aan

from utils.log import log
import utils.log

# Import log from MPD causes static variables to be loaded (e.g. VECTORS)
# Instead I copied the log function
# from eval.mpd.mpd import log


DEBUG_LIMIT = 5000
PAPER_INFO = ['title', 'venue', 'author']
# Metadata to use

print("Loading keyed vectors")
VECTORS = KeyedVectors.load_word2vec_format(str(W2V_PATH), binary=W2V_IS_BINARY)
print("Done")

# Hyperparameters
AE_PARAMS = {
    'n_code': 50,
    'n_epochs': 20,
    #    'embedding': VECTORS,
    'batch_size': 5000,
    'n_hidden': 240,
    'normalize_inputs': True,
}


def drop_paper_percentage(papers, percentage=0.5):
    shuffle(papers)
    count = int(len(papers) * percentage)
    if not count: return []
    papers[-count:], removed = [], papers[-count:]
    return removed


def papers_from_files(dataset, n_jobs=1, use_sdict=True, debug=False):
    """
    Loads a bunch of files into a list of papers,
    optionally sorted by id
    """
    if dataset == "acm":
        return load_acm(ACM_PATH)
    elif dataset == "cite":
        return load_citeworth(CITEWORTH_PATH, use_sdict)
    elif dataset == "cite2":
        return load_citeworth(CITE2_PATH, use_sdict)
    elif dataset == "cite5":
        return load_citeworth(CITE5_PATH, use_sdict)
    elif dataset == "cite7":
        return load_citeworth(CITE7_PATH, use_sdict)
    elif dataset == "cite5_papers":
        return load_citeworth(CITE5_PAPERS_PATH, use_sdict, False)
    elif dataset == "aan":
        return load_aan(AAN_PATH)

    it = glob.iglob(os.path.join(DBLP_PATH, '*.json'))
    if debug:
        print("Debug mode: using only two slices")
        it = itertools.islice(it, 2)
    n_jobs = int(n_jobs)
    if n_jobs == 1:
        papers = []
        for i, fpath in enumerate(it):
            papers.extend(load_dblp(fpath))
            print("\r{}".format(i + 1), end='', flush=True)
            if DEBUG_LIMIT and i > DEBUG_LIMIT:
                # Stop after `DEBUG_LIMIT` files
                # (for quick testing)
                break
        print()
    else:
        pps = Parallel(n_jobs=n_jobs, verbose=5)(delayed(load_dblp)(p) for p in it)
        papers = itertools.chain.from_iterable(pps)

    return list(papers)


def aggregate_paper_info(paper, attributes):
    acc = []
    for attribute in attributes:
        if attribute in paper:
            acc.append(paper[attribute])
    return ' '.join(acc)


def unpack_papers(papers, aggregate=None,end_year=-1):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """
    # Assume track_uri is primary key for track
    EXTENDED_DEBUGGING = True


    if aggregate is not None:
        for attr in aggregate:
            assert attr in PAPER_INFO

    bags_of_refs, ids, side_info, years, authors, venue, sections = [], [], {}, {}, {}, {}, {}

    # we need to keep track of the section title additionally
    sections_list = []

    title_cnt = author_cnt = ref_cnt = venue_cnt = one_ref_cnt = year_cnt = section_cnt = 0
    for paper in papers:

        if 0 < end_year <= int(paper['year']):
            continue

        # Extract ids
        ids.append(paper["id"])

        sections_list.append(paper["section_title"])
        # Put all ids of cited papers in here
        try:
            if not paper['is_citing_paper']:
                assert len(paper['references']) == 0
            # Can't say that because split is section_wise
            #else:
                #assert len(paper['references']) > 0

            # References may be missing
            bags_of_refs.append(paper["references"])
            if len(paper["references"]) > 0:
                ref_cnt += 1
            if len(paper["references"]) == 1:
                one_ref_cnt += 1
        except KeyError:
            bags_of_refs.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
            if paper["title"] != "":
                title_cnt += 1
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["year"]
            if paper["year"] == None:
                years[paper["id"]] = -1
            if paper["year"] != None and paper["year"] > 0:
                year_cnt += 1
        except KeyError:
            years[paper["id"]] = -1
        try:
            authors[paper["id"]] = paper["authors"]
        except KeyError:
            authors[paper["id"]] = []
        try:
            venue[paper["id"]] = paper["venue"]
        except KeyError:
            venue[paper["id"]] = ""
        try:
            sections[paper["id"]] = paper["section_title"]
        except KeyError:
            sections[paper["id"]] = []

        try:
            if len(paper["authors"]) > 0:
                author_cnt += 1
        except KeyError:
            pass
        try:
            if len(paper["venue"]) > 0:
                venue_cnt += 1
        except KeyError:
            pass

        try:
            if len(paper["section_title"]) > 0:
                section_cnt += 1
        except KeyError:
            pass

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_paper_info = aggregate_paper_info(paper, aggregate)
            side_info[paper["id"]] += ' ' + aggregated_paper_info

    log(
        "Metadata-fields' frequencies: references={}, title={}, authors={}, venue={}, year={}, sections={} one-reference={}"
        .format(ref_cnt / len(papers), title_cnt / len(papers), author_cnt / len(papers), venue_cnt / len(papers),
                year_cnt / len(papers), section_cnt / len(papers), one_ref_cnt / len(papers)))

    # bag_of_refs and ids should have corresponding indices
    # In side info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_refs, ids, sections_list, {"title": side_info, "year": years, "author": authors, "venue": venue,
                               "section_title": sections}


def main(year, dataset, min_count=None, outfile=None, drop=1,
         baselines=False,
         autoencoders=False,
         conditioned_autoencoders=False,
         all_metadata=True,
         use_section=False,
         only_section=False,
         use_sdict=True,
         n_code=50,
         n_hidden=100,
         val_year=-1,
         end_year=-1,
         eval_each=False):
    """ Main function for training and evaluating AAE methods on DBLP data """

    assert baselines or autoencoders or conditioned_autoencoders, "Please specify what to run"

    #AE_PARAMS['n_code'] = n_code
    #AE_PARAMS['n_hidden'] = n_hidden

    if all_metadata:
        # V2 - all metadata
        CONDITIONS = ConditionList([
            ('title', PretrainedWordEmbeddingCondition(VECTORS)),
            ('venue', PretrainedWordEmbeddingCondition(VECTORS)),
            ('author', CategoricalCondition(embedding_dim=32, reduce="sum",  # vocab_size=0.01,
                                            sparse=False, embedding_on_gpu=True))
        ])
    elif not use_section and not only_section:
        # V1 - only title metadata
        CONDITIONS = ConditionList([('title', PretrainedWordEmbeddingCondition(VECTORS))])
    elif only_section:
        CONDITIONS = ConditionList([ ('section_title', CategoricalCondition(embedding_dim=32, reduce='sum', sparse=False, embedding_on_gpu=True))])
    else:
        CONDITIONS = ConditionList([
            ('title', PretrainedWordEmbeddingCondition(VECTORS)),
            # ('section_title', PretrainedWordEmbeddingCondition(VECTORS))
            ('section_title', CategoricalCondition(embedding_dim=32, reduce='sum', sparse=False, embedding_on_gpu=True))
        ])
    #### CONDITOINS defined

    ALL_MODELS = []

    if baselines:
        # Models without metadata
        BASELINES = [
            #BM25Baseline(),
            #RandomBaseline(),
            MostPopular(),
            Countbased()
            # SVDRecommender(1000, use_title=False)
        ]

        ALL_MODELS += BASELINES

        if not all_metadata and False:
            # SVD can use only titles not generic conditions
            ALL_MODELS += [SVDRecommender(1000, use_title=True)]

    if autoencoders:
        AUTOENCODERS = [
            #AAERecommender(adversarial=False,
            #               conditions=None,
            #               lr=0.001,
            #               **AE_PARAMS),
            AAERecommender(adversarial=True,
                           conditions=None,
                           gen_lr=0.001,
                           reg_lr=0.001,
                           **AE_PARAMS)
            #VAERecommender(conditions=None, **AE_PARAMS),
            #DAERecommender(conditions=None, **AE_PARAMS)
        ]
        ALL_MODELS += AUTOENCODERS

    if conditioned_autoencoders or use_section:
        # Model with metadata (metadata used as set in CONDITIONS above)
        CONDITIONED_AUTOENCODERS = [
            # AAERecommender(adversarial=False,
            #               conditions=CONDITIONS,
            #               lr=0.001,
            #               **AE_PARAMS),
            AAERecommender(adversarial=True,
                           conditions=CONDITIONS,
                           gen_lr=0.001,
                           reg_lr=0.001,
                           **AE_PARAMS)
            # DecodingRecommender(CONDITIONS,
            #                    n_epochs=100, batch_size=1000, optimizer='adam',
            #                    n_hidden=100, lr=0.001, verbose=True),
            # VAERecommender(conditions=CONDITIONS, **AE_PARAMS),
            # DAERecommender(conditions=CONDITIONS, **AE_PARAMS)
        ]
        ALL_MODELS += CONDITIONED_AUTOENCODERS

    print("Finished preparing models:", *ALL_MODELS, sep='\n\t')

    papers = papers_from_files(dataset, n_jobs=4)

    # removing = 0.5
    # print("Too much entries. Removing {}% of entries".format(removing*100))
    # drop_paper_percentage(papers,removing)

    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, sections, side_info = unpack_papers(papers)
    del papers
    bags = Bags(bags_of_papers, ids,sections, side_info)
    if args.compute_mi:
        from models.utils import compute_mutual_info
        print("[MI] Dataset:", dataset)
        print("[MI] min Count:", min_count)
        tmp = bags.build_vocab(min_count=min_count, max_features=None)
        mi = compute_mutual_info(tmp, conditions=None, include_labels=True,
                                 normalize=True)
        with open('mi.csv', 'a') as mifile:
            print(dataset, min_count, mi, sep=',', file=mifile)

        print("=" * 78)
        exit(0)

    log("Whole dataset:")
    log(bags)

    u = set(bags.bag_owners)
    v = set([bags.bag_owners[i] for i in range(0,len(bags.bag_owners)) if len(bags.data[i]) > 0])
    log(f"Keeping {len(u)} papers")
    log(f"Citing papers: {len(v)}")

    evaluation = Evaluation(bags, year,conditions=CONDITIONS, logfile=Path(outfile), eval_each=eval_each)

    evaluation.setup(min_count=min_count, min_elements=1, drop=drop)
    log("~ Partial List + Titles + Author + Venue", "~" * 42)

    do_grid_search = False
    if do_grid_search:
        evaluation.grid_search(batch_size=1000)
    else:
        evaluation(ALL_MODELS, batch_size=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('--val', type=int, default=-1,
                        help='First year of the validation set. If not supplied no validation set will be used')
    parser.add_argument('--end', type=int, default=-1, help='If Specified every paper of this year and newer will be dropped and not be used.')
    parser.add_argument('-d', '--dataset', type=str,
                        help="Parse the DBLP,Citeworth or ACM dataset", default="acm",
                        choices=["dblp", "acm", "cite", "cite2", "cite5","cite5_papers", "aan", "cite7"])
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=None)
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        type=str, default=None)
    parser.add_argument('-dr', '--drop', type=str,
                        help='Drop parameter', default="1")
    parser.add_argument('--code', type=int, help='number of code neurons', default=50)
    parser.add_argument('--hidden', type=int, help='number of hidden neurons', default=100)
    parser.add_argument('--compute-mi', default=False,
                        action='store_true')
    parser.add_argument('--all_metadata', default=False,
                        action='store_true')
    parser.add_argument('--baselines', default=False,
                        action='store_true')
    parser.add_argument('--autoencoders', default=False,
                        action='store_true')
    parser.add_argument('--conditioned_autoencoders', default=False,
                        action='store_true')
    parser.add_argument('--use_section', default=False, action='store_true')
    parser.add_argument('--only_section', default=False, action='store_true')
    parser.add_argument('--use_sdict', default=False, action='store_true')
    parser.add_argument('--eval_each', default=False, action='store_true')
    args = parser.parse_args()

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    utils.log.LOGFILE = Path(args.outfile)

    main(year=args.year, dataset=args.dataset, min_count=args.min_count, outfile=args.outfile, drop=drop,
         all_metadata=args.all_metadata,
         baselines=args.baselines,
         autoencoders=args.autoencoders,
         conditioned_autoencoders=args.conditioned_autoencoders,
         use_section=args.use_section,
         only_section=args.only_section,
         use_sdict=args.use_sdict,
         n_code=args.code,
         n_hidden=args.hidden,
         val_year=args.val,
         end_year=args.end,
         eval_each=args.eval_each)
