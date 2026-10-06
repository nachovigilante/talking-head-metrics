# FLAME model files

The FLAME model is distributed by the Max Planck Institute for Intelligent
Systems under its own license (non-commercial research use) and is not included
here. After registering at https://flame.is.tue.mpg.de/, download:

- FLAME 2020, unzipped so that `FLAME2020/generic_model.pkl` exists;
- `FLAME_masks.pkl` (vertex masks: lips, forehead, eye region, ...) into `FLAME_masks/`;
- `landmark_embedding.npy` into this folder.

`models/flame.py` and `models/lbs.py` come from FLAME_PyTorch and DECA and keep
the MPG license headers; they are covered by the same license as the model.
