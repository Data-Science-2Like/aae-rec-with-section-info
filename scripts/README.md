# Experiments Scripts

This directory includes the scripts used to run the experiments from our report.

## Content

- `baseline.sh`: Evaluates all baselines on our modified S2ORC dataset.
- `baseline_aan.sh`: Evaluates all baselines on the [ACL Anthology Network](https://aan.how/download/) dataset .
- `baseline_dblp.sh`: Evaluates all baselines on the [DBLP](https://www.aminer.org/citation) dataset.
- `create_candidate_pools.sh`: Creates static candidate pools for the Prefetcher+Reranker experiments.
- `prefetcher_outputs.sh`: Creates static candidate pools for the Prefetcher+Reranker experiments.
- `recommender_aan.sh`: Evaluates the AAE-Recommender on the ACL Anthology Network
- `recommender_citeworth.sh`: Evaluates the AAE-Recommender on our modified S2ORC dataset
- `recommender_dblp.sh`: Evaluates the AAE-Recommender on the DBLP dataset.
- `test_s2orc_new_split.sh`: Evaluates the AAE-Recommender using all paper beginning with 2019 as test dataset and retrain on validation split.
- `test_s2orc_new_split_no_retrain.sh`: Evaluates the AAE-Recommender using all paper beginning with 2019 as test dataset and NOT retrain on validation split.
- `test_s2orc_old_split.sh`: Evaluates the AAE-Recommender using all paper beginning with 2020 as test dataset and retrain on validation split.
- `test_s2orc_old_split_no_retrain.sh`: Evaluates the AAE-Recommender using all paper beginning with 2020 as test dataset and NOT retrain on validation split.