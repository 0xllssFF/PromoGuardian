# PromoGuardian
## Environment Setup
Install the base environment
```
# conda env create -f env.yml -n promoguardian
```

## Data Download
Download the data from the following link and put it in the same directory as the code.

https://osf.io/rasje/?view_only=671050154acf4c0fa6b86a9337e74c2c
## Data Preprocess 
Generate the edge embedding using the pretrained TransR model.
```
# python edge_feature_KG.py
```
### Test Detection Model
Run the following command to test the detection model.
```
# python test.py
```
