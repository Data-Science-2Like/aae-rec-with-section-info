from pathlib import Path

# Set to the word2vec-Google-News-corpus file
# lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
W2V_PATH = Path("./vectors/GoogleNews-vectors-negative300.bin.gz")
W2V_IS_BINARY = True


# Set to a folder containing both ACM and DBLP datasets
# lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
DATA_PATH = Path("./aminer/")