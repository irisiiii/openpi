import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_jaka_example() -> dict:
    """Creates a random input example for the Jaka policy."""
    return {
        "state": np.random.rand(8),
        "wrist_image_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    """解析图像数据，支持多种输入格式
    
    支持的格式：
    1. numpy数组（推荐，服务器端已经处理了msgpack转换）
    2. 普通列表/数组
    
    注意：msgpack格式的转换现在在服务器端处理，这里不再需要处理字典格式
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 转换为numpy数组
    image = np.asarray(image)
    logger.info(f"[_parse_image] 输入: shape={image.shape}, dtype={image.dtype}")
    
    # 如果是浮点数，转换为uint8
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # 如果是CHW格式，转换为HWC
    if len(image.shape) >= 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
        logger.info(f"[_parse_image] 转换CHW->HWC: shape={image.shape}")
    
    return image


@dataclasses.dataclass(frozen=True)
class JakaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. 
    It is used for both training and inference.
    
    For Jaka robot:
    - state: 8 dimensions (7 joints + 1 gripper)
    - images: wrist_image_left and top camera
    - actions: 8 dimensions (7 joints + 1 gripper)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W), this gets skipped for policy inference
        # After repack transform, keys are: wrist_image_left, top, state, actions, prompt
        wrist_image = _parse_image(data["wrist_image_left"])
        top_image = _parse_image(data["top"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                # Use top camera as base_0_rgb (third-person view)
                "base_0_rgb": top_image,
                # Use wrist camera as left_wrist_0_rgb
                "left_wrist_0_rgb": wrist_image,
                # Pad right wrist image with zeros (Jaka only has left wrist camera)
                "right_wrist_0_rgb": np.zeros_like(top_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class JakaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. 
    It is used for inference only.
    
    For Jaka robot: 8 action dimensions (7 joints + 1 gripper)
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 actions (7 joints + 1 gripper)
        # The model outputs may be padded to a higher dimension, so we need to slice
        return {"actions": np.asarray(data["actions"][:, :8])}

