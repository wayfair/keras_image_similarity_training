# Keras Image Similarity Training
Train a convolutional neural network to determine content-based similarity between images.
This is done with a siamese neural network as shown
[here](https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py).
The model learns from labeled images of similar and dissimiar pairs. The model's objective is to
embed similar pairs nearby and dissimilar pairs far apart. This property of the latent space means
kNN searches can find similar images. This idea is based on the paper found
[here](https://cs.nyu.edu/~sumit/research/assets/cvpr06.pdf).

## Requirements
- [Docker](https://docs.docker.com/install/)
- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker) *If using GPUs*

## Labeled Data
For both training and indexing, labeled data will be needed.
This data needed is multiple images of each unique item. Create a JSON file
such as the one seen below. The key of top level items should be
the `item_id`. Each value should have an `images` array, which contains
data on each image for that item. Optionally, you can also provide `labels`
for each `item_id`, where two items sharing some label will not be
considered dissimilar.
```json
{
	"item_id_1": {
		"images": [
			{
				"filename": "relative/path/to/item_1_1.jpg"
			},
			{
				"filename": "relative/path/to/item_1_2.jpg"
			}
		],
		"labels": ["red", "pink"]
	},
	"item_id_2": {
		"images": [
			{
				"filename": "relative/path/to/item_2_1.jpg"
			},
			{
				"filename": "relative/path/to/item_2_2.jpg"
			}
		],
		"labels": ["blue"]
	}
}
```

## Training
For training a model, you will definitely need a GPU. If you do not have one,
then we suggest only using a pretrained model provided by Keras's API.

### Notebook
We provide a Jupyter notebook that will walk you through how to train a siamese network.
Note you will need a machine with an Nvidia GPU here.
```bash
DATA=/path/to/images/and/label/files make notebook
```

### Exporting Model
If you trained a model, run the following
```bash
make bash-cpu
python utilities.py --export savedmodel --keras-model checkpoints/file_saved_by_notebook.hdf5
```
Else you can use Google's pretrained model on classification
```bash
make bash-cpu
python utilities.py --export savedmodel
```

## Indexing
Images need to be embedded and indexed for fast kNN search.

GPU and a trained model
```bash
DATA=/path/to/images/and/label/files make bash-gpu
python utilities.py --export balltree \
    --keras-model checkpoints/file_saved_by_notebook.hdf5 \
    --labeled-data /data/path_to_labeled_images_file.json \
    --image-dir /data/whereever_the_base_image_dir_is_mounted
```

GPU and Google's pretrained model
```bash
DATA=/path/to/images/and/label/files make bash-gpu
python utilities.py --export balltree \
    --labeled-data /data/path_to_labeled_images_file.json \
    --image-dir /data/whereever_the_base_image_dir_is_mounted
```

CPU and Google's pretrained model
```bash
DATA=/path/to/images/and/label/files make bash-cpu
python utilities.py --export balltree \
    --labeled-data /data/path_to_labeled_images_file.json \
    --image-dir /data/whereever_the_base_image_dir_is_mounted
```