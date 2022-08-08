import joblib
import glob
import json

DATA_DIR = 'C:/Users/Simon/Desktop/prefetcher_outputs'

RERANKER_FILE = 'C:/Users/Simon/Desktop/reranker_test_citing_ids.joblib'
reranker_citing_ids = joblib.load(RERANKER_FILE)

def load_papers_info(file):
    papers_info = dict()

    with open(file) as f:
        for line in f:
            entry = json.loads(line.strip())

            keep_keys = ['paper_title', 'paper_abstract', 'paper_year']

            papers_info[entry['paper_id']] = dict([(k, v) for k, v in entry.items() if k in keep_keys])

    return papers_info

paper_info = load_papers_info('papers.jsonl')


def do_joblib_check(data: dict):
    citing_papers = set()

    for k in data.keys():
        citing_papers.add(k)

    too_much_ids = citing_papers - reranker_citing_ids

    missing_ids = reranker_citing_ids - citing_papers


    print(f"Citing papers {len(citing_papers)}")
    print(f"Missing papers {len(missing_ids)}")
    #for id in missing_ids:
    #    print(paper_info[id])
    print(f"To Much papers {len(too_much_ids)}")
    #for id in too_much_ids:
    #    print(paper_info[id])


if __name__ == '__main__':
    job_files= glob.glob(f'{DATA_DIR}/*.joblib')
    print(f"Found {len(job_files)} joblib files")

    for file in job_files:
        data = joblib.load(file)
        do_joblib_check(data)