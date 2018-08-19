import json
import logging
import os
import pickle
from itertools import zip_longest

import click
import numpy as np
from tensorflow.saved_model import signature_def_utils, builder, tag_constants, signature_constants
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.neighbors import BallTree
from PIL import Image


logger = logging.getLogger('keras_training')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(ch)

IMAGE_DIMENSIONS = (299, 299, 3)
SEARCH_INDEX_FILENAME = 'search_index.pkl'


def get_default_inception_model():
    input = Input(shape=IMAGE_DIMENSIONS)
    return InceptionResNetV2(weights='imagenet', input_tensor=input)


def get_default_inception_model_signature():
    model = get_default_inception_model()

    signature = signature_def_utils.predict_signature_def(
        inputs={'input': model.input},
        outputs={
            'embedding': model.layers[-1].input,
            'softmax': model.layers[-1].output
        }
    )
    return signature


def export_savedmodel_for_tensorflow_serving(basepath, version, signature_name, signature):
    """Exports a model signature to a SavedModel format

    Args:
        basepath (str): Path to store SavedModel files
        version (int): Version of the model being exported
        signature_name (str): Name of signature to request in Tensorflow Serving
        signature (signature_def): Signature to make available to Tensorflow Serving
    """
    save_path = os.path.join(basepath, str(version))
    model_builder = builder.SavedModelBuilder(save_path)
    model_builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
            signature_name: signature
        }
    )
    model_builder.save()


def convert_image_to_square_rgb(image, image_dims=IMAGE_DIMENSIONS[:2]):
    """Takes an image, converts it to RGB, pad to a square image, resizes,
    and converts to numpy array.

    Args:
        image (PIL.Image): Image to process

    Returns:
        PIL.Image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    if width != height:
        max_dimension = max(image.size)
        square_image = Image.new('RGB', (max_dimension, max_dimension), 'white')
        paste_location = ((max_dimension - width) // 2, (max_dimension - height) // 2)
        square_image.paste(image, paste_location)
        image = square_image

    if image.size != image_dims:
        image = image.resize(image_dims, Image.ANTIALIAS)
        
    return np.array(image)


def preprocess_image_inception_keras(image_matrix):
    X = np.array(image_matrix, dtype=np.float32)
    return preprocess_input(X)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def image_batches(rows, batch_size, image_dir=None):
    """A generator that yields a batch of images from a flat list of image
    data.

    Args:
        rows (List): List of (metadata, filename) pairs
        batch_size (int): The max size of each batch of images
        image_dir (Optional[str]): The root dir for where images are stored,
                                   use if filename in rows is relative

    Returns:
        tuple(dict, ndarray)
        """
    for batch in grouper(batch_size, rows, (None, None)):
        images = []
        payloads = []
        for metadata, filename in batch:
            if filename:
                if image_dir:
                    filename = os.path.join(image_dir, filename)
                image = Image.open(filename)
                x = convert_image_to_square_rgb(image)
                images.append(x)
                payloads.append(metadata)
            else:
                break
        yield (payloads, preprocess_image_inception_keras(images))


@click.command()
@click.option('--export', type=click.Choice(['savedmodel', 'balltree']))
@click.option('--keras-model')
@click.option('--labeled-data')
@click.option('--image-dir')
@click.option('--batch-size', default=1024)
def cli(export, keras_model, labeled_data, image_dir, batch_size):
    if export == 'savedmodel' and keras_model:
        logger.info('Starting export of {} to SavedModel'.format(keras_model))
        model = load_model(
            keras_model,
            custom_objects={'contrastive_loss': contrastive_loss}
        )
        embedder = model.get_layer('embedding_model')
        image_tensor = embedder.inputs[0]
        embedding_tensor = embedder.outputs[0]
        signature = signature_def_utils.predict_signature_def(
            inputs={'input': image_tensor},
            outputs={
                'embedding': embedding_tensor,
            }
        )
        export_savedmodel_for_tensorflow_serving('/app/savedmodels/inception_resnet_v2', 2, 'prediction', signature)
    elif export == 'savedmodel' and not keras_model:
        logger.info('Starting export of default SavedModel')
        signature = get_default_inception_model_signature()
        export_savedmodel_for_tensorflow_serving('/app/savedmodels/inception_resnet_v2', 1, 'prediction', signature)
    elif export == 'balltree':
        if keras_model:
            logger.info('Starting BallTree creation for {}'.format(keras_model))
            model = load_model(
                keras_model,
                custom_objects={'contrastive_loss': contrastive_loss}
            )
            model = model.get_layer('embedding_model')
        else:
            logger.info('Starting BallTree creation for default model')
            inception = get_default_inception_model()
            model = Model(
                inputs=[inception.input],
                outputs=[inception.layers[-1].input, inception.layers[-1].output],
                name='embedding_model'
            )

        with open(labeled_data) as f:
            item_image_data = json.loads(f.read())

        rows = []
        for item_id, data in item_image_data.items():
            for image in data['images']:
                # Edit here if you want more data in the payload for each image
                payload = {
                    'item_id': item_id,
                    'filename': image['filename']
                }
                row = (payload, image['filename'])
                rows.append(row)

        logger.info('Embedding {} images'.format(len(rows)))
        payloads = []
        embeddings = []
        num_batches = len(rows) // batch_size
        for i, (payload_batch, images) in enumerate(image_batches(rows, batch_size, image_dir)):
            print(' {}/{} batches done'.format(i, num_batches), end='\r')
            outputs = model.predict(images)
            if type(outputs) == list:
                embedding, softmax = outputs
            elif type(outputs) == np.ndarray:
                embedding = outputs
            payloads.extend(payload_batch)
            embeddings.append(embedding)
        print()

        logger.info('Making ball tree')
        X = np.vstack(embeddings)
        ball_tree = BallTree(X, leaf_size=40)
        index_data = {
            'metadata': payloads,
            'index': ball_tree
        }

        logger.info('Pickling to {}'.format(SEARCH_INDEX_FILENAME))
        with open(SEARCH_INDEX_FILENAME, 'wb') as f:
            pickle.dump(index_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cli()
