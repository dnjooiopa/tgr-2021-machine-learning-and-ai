import time
import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pipeline_path = "./output/pipeline.config"
checkpoint_path = "./output/checkpoints"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path, 'ckpt-1')).expect_partial()

@tf.function
def detect(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
