import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from orange_cv.data import scan_dataset,validate_orange_labels,split_dataset,audit
from orange_cv.features import features_from_bgr
from orange_cv.cnn import SmallCNN
import torch

class PipelineTests(unittest.TestCase):
    def test_feature_vector_finite(self):
        img=np.random.default_rng(0).integers(0,256,(100,100,3),dtype=np.uint8)
        vec=features_from_bgr(img)
        self.assertEqual(vec.shape,(157,))
        self.assertTrue(np.isfinite(vec).all())
    def test_cnn_output(self):
        model=SmallCNN(4)
        self.assertEqual(tuple(model(torch.randn(2,3,128,128)).shape),(2,4))
    def test_accept_oranges(self):
        with tempfile.TemporaryDirectory() as t:
            for cls in ['blackspot','canker','fresh','grenning']:
                p=Path(t)/cls;p.mkdir()
                for j in range(6): Image.new('RGB',(110,110),(90,150,10)).save(p/f'{j}.png')
            df=scan_dataset(t)
            validate_orange_labels(df)
            self.assertEqual(len(df),24)

if __name__=='__main__':unittest.main()
