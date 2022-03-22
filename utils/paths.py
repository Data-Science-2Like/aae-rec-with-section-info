from pathlib import Path

# Set to the word2vec-Google-News-corpus file
# lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
W2V_PATH = Path("./vectors/GoogleNews-vectors-negative300.bin.gz")
W2V_IS_BINARY = True


# Set to a folder containing both ACM and DBLP datasets
# lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
ACM_PATH = Path("./aminer/acm.txt")

DBLP_PATH = Path("./aminer/dblp-ref/")

CITEWORTH_PATH = Path("./citeworth/aae_recommender_with_section_info_v1.jsonl")

CITE2_PATH = Path("./citeworth/aae_recommender_with_section_info_v2.jsonl")

AAN_PATH = Path("./aan/2014/")