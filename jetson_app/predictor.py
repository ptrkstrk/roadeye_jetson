from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import *
from detectron2.checkpoint import *
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T, torch

class MTSDPredictor(DefaultPredictor):

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ('RGB', 'BGR'), self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == 'RGB':
                print('RGB')
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype('float32').transpose(2, 0, 1))
            inputs = {'image':image, 
             'height':height,  'width':width}
            predictions = self.model([inputs])[0]
            return predictions
