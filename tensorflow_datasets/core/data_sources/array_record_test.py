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

"""grain tests."""

from unittest import mock

import pytest
from tensorflow_datasets.core import dataset_info as dataset_info_lib
from tensorflow_datasets.core import decode
from tensorflow_datasets.core import file_adapters
from tensorflow_datasets.core.data_sources import array_record
from tensorflow_datasets.core.utils import shard_utils

from array_record.python import array_record_data_source


_FILE_INSTRUCTIONS = [
    shard_utils.FileInstruction(
        'my_file-000-of-003', skip=0, take=12, examples_in_shard=12
    ),
    shard_utils.FileInstruction(
        'my_file-001-of-003', skip=2, take=9, examples_in_shard=11
    ),
    shard_utils.FileInstruction(
        'my_file-002-of-003', skip=0, take=4, examples_in_shard=4
    ),
]


def create_dataset_info():
  dataset_info = mock.create_autospec(dataset_info_lib.DatasetInfo)
  dataset_info.file_format = file_adapters.FileFormat.ARRAY_RECORD
  dataset_info.splits = {'train': mock.MagicMock()}
  dataset_info.splits['train'].file_instructions = _FILE_INSTRUCTIONS
  dataset_info.name = 'dataset_name'
  return dataset_info


@pytest.mark.parametrize(
    ['file_format'],
    [
        (file_adapters.FileFormat.RIEGELI,),
        (file_adapters.FileFormat.SSTABLE,),
        (file_adapters.FileFormat.TFRECORD,),
    ],
)
def test_unsupported_file_formats_raise_error(file_format):
  dataset_info = create_dataset_info()
  dataset_info.file_format = file_format
  with pytest.raises(
      NotImplementedError,
      match='No random access data source for file format',
  ):
    array_record.ArrayRecordDataSource(dataset_info, split='train')


def test_missing_split_raises_error():
  dataset_info = create_dataset_info()
  with pytest.raises(
      IndexError,
      match='Split "doesnotexist" is not in',
  ):
    array_record.ArrayRecordDataSource(dataset_info, split='doesnotexist')


def test_array_record_file_format_delegates_to_array_record_data_source():
  dataset_info = create_dataset_info()
  dataset_info.features = mock.MagicMock()
  deserialize_example_mock = mock.MagicMock(return_value='decoded_example')
  dataset_info.features.deserialize_example = deserialize_example_mock
  with mock.patch.object(
      array_record_data_source, 'ArrayRecordDataSource'
  ) as array_record_data_source_mock:
    array_record.ArrayRecordDataSource(dataset_info, split='train')
    array_record_data_source_mock.assert_called_once_with(_FILE_INSTRUCTIONS)


def test_repr_returns_meaningful_string_without_decoders():
  dataset_info = create_dataset_info()
  with mock.patch.object(array_record_data_source, 'ArrayRecordDataSource'):
    source = array_record.ArrayRecordDataSource(dataset_info, split='train')
    assert (
        repr(source)
        == "GrainDataSource(name=dataset_name, split='train', decoders=None)"
    )


def test_repr_returns_meaningful_string_with_decoders():
  dataset_info = create_dataset_info()
  with mock.patch.object(array_record_data_source, 'ArrayRecordDataSource'):
    source = array_record.ArrayRecordDataSource(
        dataset_info,
        split='train',
        decoders={'my_feature': decode.SkipDecoding()},
    )
    assert (
        repr(source)
        == 'GrainDataSource(name=dataset_name,'
        " split='train', decoders={'my_feature': <class"
        " 'tensorflow_datasets.core.decode.base.SkipDecoding'>})"
    )
