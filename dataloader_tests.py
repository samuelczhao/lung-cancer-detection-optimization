import unittest
from dataloader_cropped_mass import *

class TestDataLoadingMethods(unittest.TestCase):
    def test_async_data_loader(self):
        data_gen = async_data_loader(size=(226, 226), start=0, batch_size=32)
        for i in range(100):
            img, lbl = next(data_gen)
            self.assertTrue(img.shape == (32, 226, 226, 1),
                    msg="Wrong image shape: " + str(img.shape))
            self.assertTrue(lbl.shape == (32,),
                    msg="Wrong label shape: " + str(lbl.shape))
