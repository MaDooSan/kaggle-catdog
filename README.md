# **Deep Learning with Kaggle Cats and Dogs Dataset on MSI**

## I. Training Kaggle Cats and Dogs Dataset on Caffe (Python)

1. Load modules and environment
```
module load caffe/1.0
source activate caffe1.0_lite
```
Tip: if you plan to install new packages, you can use a clone environment:

`conda create --clone <source env> --name <clone env>`

2. Make data directory
```
cd kaggle-catdog
mkdir input
cd input
```
3. Download dataset and extract; if your data is stored elsewhere, use symbolic links:

`ln -s <target_dir> <link_name>`

4. Create LMDB creatdataset to :

1) run histogram-equalization on all training images, resize all training images to a 227x227 format.
2) divide the training data into 2 sets: One for training (5/6 of images) and the other for validation (1/6 of images)
3) Store the training and validation in 2 LMDB databases (train and val)
```
cd code
python create_lmdb.py
```
5. Compute image-mean (used to make training data zero-mean)

`compute_image_mean.bin -backend=lmdb ../input/train_lmdb ../input/mean.binaryproto`

6. Pick a model definition (we used the BVLC Caffenet model) and edit the following on the train prototxt model files:

1) pathname for input data and mean image
2) change number of outputs from 1000 to 2 (as the original model was trained to classify 1000 classes)

7. Update the solver definitions with the new pathnames for `net` and `snapshot`

Note: Change solver parameters as needed; by default the solver computes the accuracy of the model using the validation set every 1000 iterations; the optimization process will run for a maximum of 20000 iterations, and will take a snapshot of the trained model every 5000 iterations.

8. Train!

`caffe.bin train --solver ../caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee ../caffe_models/caffe_model_1/model_1_train.log`

Notes: We use "tee" to redirect output to a log file (as shown above)
If for some reason, training quits (maybe you exceeded walltime limit or something else failed), you can use the snapshots (saved as .solverstate files) to resume training

`caffe.bin train --solver ../caffe_models/caffe_model_1/solver_1.prototxt --snapshot <solverstate_file>`
