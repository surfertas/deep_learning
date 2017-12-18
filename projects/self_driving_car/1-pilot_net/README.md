Notes
---

12/18/2017
First pass implementation of PilotNets pipeline complete. Training and
validation is done by running `python pilot_net.py` and evaluation based on RMSE
with visualization can be executed by running `python evaluate.py`. Evaluation
is using the validation set just to get the pipeline in place. I need to
download the test data and incorporate into the pipeline. Training only on Bag 4
results in the model learning the average. No significant pre-processing nor
data augmentation has been incorporated. Next steps include increasing the
training data by incorporating other bags (Bag 5), shuffling data, flipping, and
micro rotations to replicate jitter.
