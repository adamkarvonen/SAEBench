## RAVEL Benchmark, adapted for SAEs

#### Task 
RAVEL quantifies feature disentanglement. Given a dataset of entities (eg. cities) with multiple attributes (eg. country, language) we score an SAE's ability to have a precise causal effect on one of the attributes while leaving other attributes unaffected. The current form computes the disentanglement of two attributes `A` and `B` from a single file.

#### Method


#### Implementation
The scoring consists of three steps:
1. Create a `RAVELInstance`, a dataset of Entity-Attribute pairs filterd to only contain pairs the model actually knows. The `RAVELInstance.prompts` contain tokenized prompts and more metadata. See `instance.py` and `generation.py`.
2. Select attribute-sprecific SAE latens with cosine similarity to DAS.
3. Compute cause and isolation scores by intervening on attribute specific features
    - Cause evaluation: High accuracy if intervening with A_features is successful on base_A_template, ie. source_A_attribute_value is generated.
    - Isolation evaluation: High accuracy if intervening with B_features is unsuccessful on base_A_template, ie. base_A_attribute is generated regardless of intervention.
    - disentanglement_score is the mean: D = (cause_A[t] + cause_B[t] + isolation_AtoB[t] + isolation_BtoA[t]) / 4
    - see `intervention.py`


#### Steps
1. Implement DAS for neurons
    - What hyperparameters does DAS have? How do the RAVEL papers choose?
2. Check cosine sim of SAE features with