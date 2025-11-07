import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_fold_cloth_example() -> dict:
    """Creates a random input example for the Fold Cloth policy (Piper dual-arm robot)."""
    return {
        "state": np.random.rand(14),  # 左臂6关节+夹爪 + 右臂6关节+夹爪
        "wrist_image_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrist_image_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "low": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Fold the shirt.",
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
class FoldClothInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. 
    It is used for both training and inference.
    
    For Piper dual-arm robot (Fold Cloth task):
    - state: 14 dimensions (left arm 6 joints + gripper + right arm 6 joints + gripper)
    - images: wrist_image_left, wrist_image_right, low (overhead camera)
    - actions: 14 dimensions (left arm 6 joints + gripper + right arm 6 joints + gripper)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W), this gets skipped for policy inference
        # After repack transform, keys are: wrist_image_left, wrist_image_right, low, state, actions, prompt
        wrist_image_left = _parse_image(data["wrist_image_left"])
        wrist_image_right = _parse_image(data["wrist_image_right"])
        low_image = _parse_image(data["low"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                # Use low camera as base_0_rgb (overhead/third-person view)
                "base_0_rgb": low_image,
                # Use left wrist camera
                "left_wrist_0_rgb": wrist_image_left,
                # Use right wrist camera
                "right_wrist_0_rgb": wrist_image_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
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
class FoldClothOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. 
    It is used for inference only.
    
    For Piper dual-arm robot: 14 action dimensions (left arm 6 joints + gripper + right arm 6 joints + gripper)
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 actions (left arm + right arm)
        # The model outputs may be padded to a higher dimension, so we need to slice
        return {"actions": np.asarray(data["actions"][:, :14])}

