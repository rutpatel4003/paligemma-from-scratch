from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
  # The input text is tokenized normally
  # A <bos> token is added at the beginning, and an additional new line token (\n) is appended as seen in the hugging face implementation (usually it is tokenized separately)
  # The tokenized text is also prefixed with a fixed number of <image> tokens.

  return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(image, size, resample=None, reducing_gap=None):
  height, width = size
  resized_image = image.resize(
    (width, height), resample=resample, reducing_gap=reducing_gap
  )

  return resized_image



def rescale(image, scale, dtype=np.float32):
  rescaled_image = image*scale
  rescaled_image = rescaled_image.astype(dtype)
  return rescaled_image


def normalize(image, mean, std):
  mean = np.array(mean, dtype=image.dtype)
  std = np.array(std, dtype=image.dtype)
  image = (image - mean) / std
  return image

def process_images(
    images, 
    size=None,
    resample=None,
    rescale_factor=None,
    image_mean=None,
    image_std=None
):
  height, width = size[0], size[1]
  images = [
    resize(image=image, size=(height, width), resample=resample) for image in images
  ]

  images = [np.array(image) for image in images]

  images = [rescale(image, scale=rescale_factor) for image in images]

  images = [normalize(image, mean=image_mean, std=image_std) for image in images]

  #H, w, c --> c, h, w
  images = [image.transpose(2, 0, 1) for image in images]

  return images

class PaliGemmaProcessor:
  IMAGE_TOKEN = "<image>"

  def __init__(self, tokenizer, num_image_tokens, image_size):
    super().__init__()

    self.image_seq_length = num_image_tokens
    self.image_size = image_size

    tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
    tokenizer.add_special_tokens(tokens_to_add)
    EXTRA_TOKENS = [
      f"<loc{i:04d}>" for i in range(1024)
    ] # Tokens used for object detection (bounding boxes)

    EXTRA_TOKENS += [
      f"<seg{i:04d}" for i in range(128)
    ] # Tokens for object segmentation

    tokenizer.add_tokens(EXTRA_TOKENS)

    self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    self.tokenizer = tokenizer

  def __call__(self, text, images, padding: str = "longest", truncation: bool = True):
    assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

    pixel_values = process_images(
      images,
      size = ((self.image_size, self.image_size)),
      resample = Image.Resampling.BICUBIC,
      rescale_factor = 1 / 255.0,
      image_mean = IMAGENET_STANDARD_MEAN,
      image_std = IMAGENET_STANDARD_STD,
    )

    # Convert list of np arrays to a single np array
    pixel_values = np.stack(pixel_values, axis=0)
    pixel_values = torch.tensor(pixel_values)

    input_strings = [
      add_image_tokens_to_prompt(
        prefix_prompt = prompt,
        bos_token = self.tokenizer.bos_token,
        image_seq_len = self.image_seq_length, 
        image_token = self.IMAGE_TOKEN
      )
      for prompt in text
    ]

    inputs = self.tokenizer(
      input_strings,
      return_tensors='pt',
      padding=padding,
      truncation=truncation
    )
    
    return_data = {"pixel_values": pixel_values, **inputs}

    return return_data


 