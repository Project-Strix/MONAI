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

from monai.data import Dataset
from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstD, FromMetaTensord, RandLazyLoadImageD, SaveImageD
from monai.transforms.meta_utility.dictionary import ToMetaTensord
from monai.utils import optional_import
from tests.utils import assert_allclose
from monai.data import NibabelLazyReader

KEYS = ["image", "label", "extra"]

TEST_CASE_1 = [{"keys": KEYS, "roi_size": (5,5,5)}, (5, 5, 5)]


TESTS_META = []
for track_meta in (False, True):
    TESTS_META.append([{"keys": KEYS, "roi_size": (5,5,5)}, (5, 5, 5), track_meta])


class TestLoadImaged(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, input_param, expected_shape):
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        test_data = {}
        with tempfile.TemporaryDirectory() as tempdir:
            for key in KEYS:
                nib.save(test_image, os.path.join(tempdir, key + ".nii.gz"))
                test_data.update({key: os.path.join(tempdir, key + ".nii.gz")})
            result = RandLazyLoadImageD(image_only=True, **input_param)(test_data)

        for key in KEYS:
            self.assertTupleEqual(result[key].shape, expected_shape)

    def test_channel_dim(self):
        spatial_size = (32, 64, 3, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            nib.save(nib.Nifti1Image(test_image, affine=np.eye(4)), filename)

            loader = RandLazyLoadImageD(keys="img", roi_size=(5,5,5), image_only=True)
            loader.register(NibabelLazyReader(channel_dim=2))
            t = Compose([EnsureChannelFirstD("img"), FromMetaTensord("img")])
            img_dict = loader({"img": filename})
            result = t(img_dict)
            self.assertTupleEqual(result["img"].shape, (3, 5, 5, 5))

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            nib.save(nib.Nifti1Image(test_image, affine=np.eye(4)), filename)

            loader = RandLazyLoadImageD(keys="img", roi_size=(5,5,5), image_only=True, channel_dim=2)
            # loader.register(NibabelLazyReader(channel_dim=2))
            t = Compose([EnsureChannelFirstD("img"), FromMetaTensord("img")])
            img_dict = loader({"img": filename})
            result = t(img_dict)
            self.assertTupleEqual(result["img"].shape, (3, 5, 5, 5))

    def test_randomize(self):
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        test_data = {}
        key = "image"
        with tempfile.TemporaryDirectory() as tempdir:
            
            nib.save(test_image, os.path.join(tempdir, key + ".nii.gz"))
            test_data = {key: os.path.join(tempdir, key + ".nii.gz")}

            loader = RandLazyLoadImageD(keys=key, image_only=True, roi_size=(10,10,10))
            ds = Dataset(data=[test_data, test_data], transform=loader)
            ds_iter = iter(ds)
            first_item = next(ds_iter)
            second_item = next(ds_iter)
            value_equal = torch.equal(first_item[key], second_item[key])
            shape_equal = (first_item[key].shape == second_item[key].shape)
            self.assertFalse(value_equal)
            self.assertTrue(shape_equal, f"loaded image shape: {first_item[key].shape}, {second_item[key].shape}")

    def test_no_file(self):
        with self.assertRaises(RuntimeError):
            RandLazyLoadImageD(keys="img", roi_size=(5,5,5), image_only=True)({"img": "unknown"})
        with self.assertRaises(RuntimeError):
            RandLazyLoadImageD(keys="img", roi_size=(5,5,5), reader="nibabelreader", image_only=True)({"img": "unknown"})


class TestLoadImagedMeta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.tmpdir = tempfile.mkdtemp()
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        cls.test_data = {}
        for key in KEYS:
            nib.save(test_image, os.path.join(cls.tmpdir, key + ".nii.gz"))
            cls.test_data.update({key: os.path.join(cls.tmpdir, key + ".nii.gz")})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand(TESTS_META)
    def test_correct(self, input_p, expected_shape, track_meta):
        set_track_meta(track_meta)
        result = RandLazyLoadImageD(image_only=True, prune_meta_pattern=".*_code$", prune_meta_sep=" ", **input_p)(
            self.test_data
        )

        # shouldn't have any extra meta data keys
        for key in KEYS:
            r = result[key]
            self.assertTupleEqual(r.shape, expected_shape)
            if track_meta:
                self.assertIsInstance(r, MetaTensor)
                self.assertTrue(hasattr(r, "affine"))
                self.assertIsInstance(r.affine, torch.Tensor)
                self.assertEqual(r.meta["space"], "RAS")
                self.assertTrue("qform_code" not in r.meta)
            else:
                self.assertIsInstance(r, torch.Tensor)
                self.assertNotIsInstance(r, MetaTensor)
                self.assertFalse(hasattr(r, "affine"))


if __name__ == "__main__":
    unittest.main()
