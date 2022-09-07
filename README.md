# AAE-Recommender with Section Information

This repository is based on https://github.com/lgalke/aae-recommender.
The implementation is adjusted to also allow for the usage of section information (e.g. section headings).

It also has been adapted to use our [Modified S2ORC Dataset](https://github.com/Data-Science-2Like/dataset-creation).
To repeat our experiments, create the modified S2ORC dataset as described in the corresponding repo and execute the scripts from the `scripts` subfolder.

## Folder structure
    ├── dataset                              # Code for loading different datasets (including our modified S2ORC)
    ├── models                         		 # Code of the original AAE-Recommender. Adapted to our needs.
    ├── scripts                              # The scripts for our different experiments
    ├── utils                                # auxilary data files
    ├── main.py                     		 # Main file which runs the experiments. Is not called directly but via the scripts in the scripts folder
    ├── statistics.py       				 # Code to generate diagrams, statistics etc.  




