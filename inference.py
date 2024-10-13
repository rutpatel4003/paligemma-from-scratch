from PIL import Image
import torch
import fire
from processing_palligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs, device):
  model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
  return model_inputs

def _sample_top_p(probs, p):
  # batch_size, vocab_size
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

  # batch_size, vocab_size
  probs_sum = torch.cumsum(probs, dim=-1)

  # subtracting probs_sort shifts the cumulative sum by 1 position to the right before masking
  mask = probs_sum - probs_sort > p

  # zero out all probabilities of tokens not selected by top p
  probs_sort[mask] = 0.0

  # redistribute the probabilities of tokens that are not selected from the top p distribution
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

  # sample token from top-p distribution
  next_token = torch.multinomial(probs_sort, num_samples=1)

  # obtaining next_token position in the vocab
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token


def get_model_inputs(processor, prompt, image_file_path, device):
  image = Image.open(image_file_path)
  images = [image]
  prompts = [prompt]
  model_inputs = processor(text=prompts, images=images)
  model_inputs = move_inputs_to_device(model_inputs, device)
  return model_inputs

def test_inference(
    model,
    processor,
    device,
    prompt,
    image_file_path, 
    max_tokens_to_generate,
    temperature,
    top_p,
    do_sample,
):
  model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
  input_ids = model_inputs['input_ids']
  attention_mask = model_inputs['attention_mask']
  pixel_values = model_inputs['pixel_values']

  kv_cache = KVCache()

  # generate until stop token
  stop_token = processor.tokenizer.eos_token_id
  generated_tokens = []

  for _ in range(max_tokens_to_generate):
    outputs = model(
      input_ids=input_ids,
      pixel_values=pixel_values,
      attention_mask=attention_mask,
      kv_cache=kv_cache
    )

    kv_cache = outputs['kv_cache']
    next_token_logits = outputs['logits'][:, -1, :]

    if do_sample:
      # sample the next token
      # apply temperature
      next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
      next_token = _sample_top_p(next_token_logits, top_p)
    
    else:
      next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    assert next_token.size() == (1, 1)

    next_token = next_token.squeeze(0)
    generated_tokens.append(next_token)
    if next_token.item() == stop_token:
      break
      
    # append next token to input since only last predicted token will be used because of kv cache
    input_ids = next_token.unsqueeze(-1)
    attention_mask = torch.cat(
      [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
    )

  generated_tokens = torch.cat(generated_tokens, dim=-1)

  # decode the generate tokens
  decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

  print(prompt + decoded)


def main (
    model_path = None,
    prompt = None,
    image_file_path = None,
    max_tokens_to_generate = 100,
    temperature = 0.8, 
    top_p = 0.9,
    do_sample = False, 
    only_cpu = False
):
  device = 'cpu'

  # if not only_cpu:
  #   if torch.cuda.is_available():
  #     device='cuda'
  #   elif torch.backends.mps.is_available():
  #     device='mps'

  print("Device in use: ", device)

  print("Loading model")
  model, tokenizer = load_hf_model(model_path, device)
  model = model.to(device).eval()

  num_image_tokens = model.config.vision_config.num_image_tokens
  image_size = model.config.vision_config.image_size
  processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

  print("Running inference")
  with torch.no_grad():
    test_inference(
      model, 
      processor,
      device,
      prompt,
      image_file_path, 
      max_tokens_to_generate,
      temperature,
      top_p, 
      do_sample
    )

if __name__ == "__main__":
  fire.Fire(main)
