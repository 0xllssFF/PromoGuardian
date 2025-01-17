# PromoGuardian
## Environment Setup
Install the base environment
```
# conda env create -f env.yml -n promoguardian
```
## Data Preprocess 
Generate the edge embedding using the results of TransR
```
# python edge_feature_KG.py
```
### Test Detection Model
Test with the newly generated edge embedding
```
# python test.py
```
