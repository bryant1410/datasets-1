# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""imagenet_pi dataset."""

import io
import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
from tensorflow_datasets.datasets.imagenet2012 import imagenet_common
import tensorflow_datasets.public_api as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for imagenet_pi dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain two files: ILSVRC2012_img_train.tar and
  ILSVRC2012_img_val.tar.
  You need to register on http://www.image-net.org/download-images in order
  to get the link to download the dataset.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    names_file = imagenet_common.label_names_file()
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'clean_label': tfds.features.ClassLabel(names_file=names_file),
            'annotator_labels': tfds.features.Tensor(
                shape=(16,), dtype=tf.int64
            ),
            'annotator_confidences': tfds.features.Tensor(
                shape=(16,), dtype=tf.float32
            ),
            'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
        }),
        supervised_keys=('image', 'annotator_labels'),
        homepage='https://github.com/google-research-datasets/imagenet_pi/',
    )

  def _get_values_from_csv(self, input_csv, to_type=int):
    """Gets the annotator labels from the csv lines list."""
    # The input_csv contains lines such as
    # n02097047_2079.JPEG,165,196,196,196,196,196,196,196
    values_dict = {}
    for cur_line in input_csv:
      elems = cur_line.split(',')
      filename = elems[0]
      values = [to_type(elem) for elem in elems[1:]]
      values_dict[filename] = values
    return values_dict

  def _get_train_annotator_labels(self, dl_manager):
    labels_path = os.path.join(dl_manager.manual_dir, 'labels/', 'train.csv')
    if not tf.io.gfile.exists(labels_path):
      raise AssertionError(
          'ImageNet-PI requires manual download of the train annotator labels. '
          'Please download them and place them into: {}'.format(labels_path)
      )
    with tf.io.gfile.GFile(labels_path, 'r') as f:
      return self._get_values_from_csv(f.read().splitlines())

  def _get_validation_annotator_labels(self, dl_manager):
    labels_path = os.path.join(
        dl_manager.manual_dir, 'labels/', 'validation.csv'
    )
    if not tf.io.gfile.exists(labels_path):
      raise AssertionError(
          'ImageNet-PI requires manual download of the validation annotator'
          ' labels. Please download them and place them into: {}'.format(
              labels_path
          )
      )
    with tf.io.gfile.GFile(labels_path, 'r') as f:
      return self._get_values_from_csv(f.read().splitlines())

  def _get_train_annotator_confidences(self, dl_manager):
    confidences_path = os.path.join(
        dl_manager.manual_dir, 'confidences/', 'train.csv'
    )
    if not tf.io.gfile.exists(confidences_path):
      raise AssertionError(
          'ImageNet-PI requires manual download of the train annotator'
          ' confidences. Please download them and place them into: {}'.format(
              confidences_path
          )
      )
    with tf.io.gfile.GFile(confidences_path, 'r') as f:
      return self._get_values_from_csv(f.read().splitlines(), to_type=float)

  def _get_validation_annotator_confidences(self, dl_manager):
    confidences_path = os.path.join(
        dl_manager.manual_dir, 'confidences/', 'validation.csv'
    )
    if not tf.io.gfile.exists(confidences_path):
      raise AssertionError(
          'ImageNet-PI requires manual download of the validation annotator'
          ' confidences. Please download them and place them into: {}'.format(
              confidences_path
          )
      )
    with tf.io.gfile.GFile(confidences_path, 'r') as f:
      return self._get_values_from_csv(f.read().splitlines(), to_type=float)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    train_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_train.tar')
    val_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_val.tar')

    if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(val_path):
      raise AssertionError(
          'ImageNet-PI requires manual download of the ImageNet2012 data. '
          'Please download the train and val set and place them into:'
          '{}, {}'.format(train_path, val_path)
      )

    return {
        'train': self._generate_examples(
            archive=dl_manager.iter_archive(train_path),
            annotator_labels=self._get_train_annotator_labels(dl_manager),
            annotator_confidences=self._get_train_annotator_confidences(
                dl_manager
            ),
            validation_labels=None,
        ),
        'validation': self._generate_examples(
            archive=dl_manager.iter_archive(val_path),
            annotator_labels=self._get_validation_annotator_labels(dl_manager),
            annotator_confidences=self._get_validation_annotator_confidences(
                dl_manager
            ),
            validation_labels=imagenet_common.get_validation_labels(val_path),
        ),
    }

  def _generate_examples(
      self,
      archive,
      annotator_labels,
      annotator_confidences,
      validation_labels=None,
  ):
    # Validation split.
    if validation_labels:
      for fname, fobj in archive:
        record = {
            'file_name': fname,
            'image': fobj,
            'clean_label': validation_labels[fname],
            'annotator_labels': annotator_labels[fname],
            'annotator_confidences': annotator_confidences[fname],
        }
        yield fname, record

    # Training split. Main archive contains archives names after a synset noun.
    # Each sub-archive contains pictures associated to that synset.
    for fname, fobj in archive:
      label = fname[:-4]  # fname is something like 'n01632458.tar'
      # TODO(b/117643231): in py3, the following lines trigger tarfile module
      # to call `fobj.seekable()`, which Gfile doesn't have. We should find an
      # alternative, as this loads ~150MB in RAM.
      fobj_mem = io.BytesIO(fobj.read())
      for image_fname, image in tfds.download.iter_archive(
          fobj_mem, tfds.download.ExtractMethod.TAR_STREAM
      ):
        record = {
            'file_name': image_fname,
            'image': image,
            'clean_label': label,
            'annotator_labels': annotator_labels[image_fname],
            'annotator_confidences': annotator_confidences[image_fname],
        }
        yield image_fname, record
