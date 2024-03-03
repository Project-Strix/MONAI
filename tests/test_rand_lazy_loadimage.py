# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized
from PIL import Image

from monai.apps import download_and_extract
from monai.data import NibabelLazyReader
from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import RandLazyLoadImage
from tests.utils import assert_allclose, skip_if_downloading_fails, testing_data_config



TEST_CASE_1 = [{"roi_size": [5,5,5]}, ["test_image.nii.gz"], (5, 5, 5)]

TEST_CASE_2 = [{"roi_size": [1,128,-1]}, ["test_image.nii.gz"], (1, 128, 128)]

TEST_CASE_3 = [{"roi_size": [5,5,5]}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], (3, 5, 5, 5)]

TEST_CASE_5 = [{"reader": NibabelLazyReader(), "roi_size": [5,5,5]}, ["test_image.nii.gz"], (5, 5, 5)]

TEST_CASE_13 = [{"reader": "nibabelreader", "channel_dim": 0, "roi_size": [5,5,5]}, "test_image.nii.gz", (3, 5, 5, 5)]

TEST_CASE_14 = [
    {"reader": "nibabelreader", "channel_dim": -1, "ensure_channel_first": True, "roi_size": (5,5,5)},
    "test_image.nii.gz",
    (5, 5, 5, 3)
]

TEST_CASE_15 = [{"reader": "nibabelreader", "channel_dim": 2, "roi_size": (5,5,5)}, "test_image.nii.gz", (5, 5, 3, 5)]

TESTS_META = []
for track_meta in (False, True):
    TESTS_META.append([{}, (5, 5, 5), track_meta])

class TestLoadImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        with skip_if_downloading_fails():
            cls.tmpdir = tempfile.mkdtemp()
            key = "DICOM_single"
            url = testing_data_config("images", key, "url")
            hash_type = testing_data_config("images", key, "hash_type")
            hash_val = testing_data_config("images", key, "hash_val")
            # download_and_extract(
            #     url=url, output_dir=cls.tmpdir, hash_val=hash_val, hash_type=hash_type, file_type="zip"
            # )
            cls.data_dir = os.path.join(cls.tmpdir, "CT_DICOM_SINGLE")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_5])
    def test_nibabel_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = RandLazyLoadImage(image_only=True, **input_param)(filenames)
            ext = "".join(Path(name).suffixes)
            self.assertEqual(result.meta["filename_or_obj"], os.path.join(tempdir, "test_image" + ext))
            self.assertEqual(result.meta["space"], "RAS")
            assert_allclose(result.affine, torch.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)


    @parameterized.expand([TEST_CASE_13, TEST_CASE_14, TEST_CASE_15])
    def test_channel_dim(self, input_param, filename, expected_shape):
        test_image = np.random.rand(*expected_shape)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)
            result = RandLazyLoadImage(image_only=True, **input_param)(filename)

        self.assertTupleEqual(
            result.shape, (3, 5, 5, 5) if input_param.get("ensure_channel_first", False) else expected_shape
        )
        self.assertEqual(result.meta["original_channel_dim"], input_param["channel_dim"])


class TestLoadImageMeta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.tmpdir = tempfile.mkdtemp()
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        nib.save(test_image, os.path.join(cls.tmpdir, "im.nii.gz"))
        cls.test_data = os.path.join(cls.tmpdir, "im.nii.gz")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand(TESTS_META)
    def test_correct(self, input_param, expected_shape, track_meta):
        set_track_meta(track_meta)
        r = RandLazyLoadImage(image_only=True, roi_size=(5,5,5), prune_meta_pattern="glmax", prune_meta_sep="%", **input_param)(self.test_data)
        self.assertTupleEqual(r.shape, expected_shape)
        if track_meta:
            self.assertIsInstance(r, MetaTensor)
            self.assertTrue(hasattr(r, "affine"))
            self.assertIsInstance(r.affine, torch.Tensor)
            self.assertTrue("glmax" not in r.meta)
        else:
            self.assertIsInstance(r, torch.Tensor)
            self.assertNotIsInstance(r, MetaTensor)
            self.assertFalse(hasattr(r, "affine"))


if __name__ == "__main__":
    unittest.main()
