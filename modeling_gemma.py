import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SigLipVisionModel

class KVCache():
  def __init__(self):
    self.key_cache = []
    self.value_cache = []

  def num_items(self):
    if len(self.key_cache) == 0:
      return 0
    else:
      # batch_size, num_heads_kv, seq_len, head_dim
      return self.key_cache[0].shape[-2]
    
  def update(self, key_states, value_states, layer_idx):
    if len(self.key_cache) <= layer_idx:
      # adds the layer's kv_cache
      self.key_cache.append(key_states)
      self.value_cache.append(value_states)
    else:
      # concatenate new keys with existing ones
      # batch_size, num_heads_kv, seq_len, head_dim (concatenating along the seq_len dimension)
      self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
      self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

    return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
  def __init__(self,
               vocab_size,
               hidden_size,
               intermediate_size, 
               num_hidden_layers,
               num_attention_heads, # num of heads for queries
               num_key_value_heads, # num of heads for key and value 
               head_dim=256,
               max_position_embeddings=8192,
               rms_norm_eps = 1e-6,
               rope_theta = 10000.0,
               attention_bias = False,
               attention_dropout = 0.0,
               pad_token_id=None,
               **kwargs
               ):
    super().__init__()

    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.num_key_value_heads = num_key_value_heads
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout
    self.pad_token_id = pad_token_id

    
    

class PaliGemmaConfig(nn.Module):
  def __init__(
    self, 
    vision_config=None,
    text_config=None,
    ignore_index=-100,
    image_token_index=256000,
    vocab_size=257152,
    projection_dim=2048,
    hidden_size=2048,
    pad_token_id=None,
    **kwargs
  ):
    super().__init__()
    self.ignore_index = ignore_index
    self.image_token_index = image_token_index
    self.projection_dim = projection_dim
    self.hidden_size = hidden_size
    self.vision_config = vision_config
    self.is_encoder_decoder = False # Only used for hugging face implementation
    self.pad_token_id = pad_token_id

    self.vision_config = SiglipVisionConfig(**vision_config)
    self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

    self.vocab_size = self.text_config.vocab_size

    self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
    self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.zeros(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  
  def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)

class GemmaMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

  def forward(self, x):
    return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))

def repeat_kv(hidden_states, n_rep):
  batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim) # None is for the new dimension
  return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)

class GemmaRotaryEmbedding(nn.Module):
  def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
    super().__init__()

    self.dim = dim # head dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base # set to 10000 in the roformer paper

    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
    self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

  @torch.no_grad()
  def forward(self, x, position_ids, seq_len=None):
    # batch_size, num_attention_heads, seq_len, head_size
    self.inv_freq.to(x.device)
    # extend inv_freq --> batch_size, head_dim // 2, 1
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    
    # batch_size, 1, seq_len
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
      # multiplying each theta by the position (argument of sine and cosine functions)
      # batch_size, head_dim//2, 1 @ batch_size, 1, seq_len --> batch_size, seq_len, head_dim // 2
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

      # batch_size, seq_le, head_dim
      emb = torch.cat((freqs, freqs), dim=-1)

      # cos, sin: batch_size, seq_len, head_dim
      cos = emb.cos()
      sin = emb.sin()

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2] # first half of last dimension
  x2 = x[..., x.shape[1] // 2 :] # second half of last dimension
  return torch.cat((-x1, x1), dim=-1)

# hugging face embedding function
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
  # add the head dimension
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)

  # apply formula 34 of roformer paper
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin) 

  return q_embed, k_embed

class GemmaAttention(nn.Module):
  def __init__(self, config, layer_idx=None):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = config.head_dim
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    assert self.hidden_size % self.num_heads == 0

    # We have smaller number of heads for kv (1 for paligemma), because of this, we get smaller projection of embeddings for each token
    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
    self.rotary_emb = GemmaRotaryEmbedding(
      self.head_dim,
      max_position_embeddings = self.max_position_embeddings,
      base = self.rope_theta
    )

  def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
    batch_size, q_len, _ = hidden_states.size()

    # batch_size, seq_len, num_heads_q * head_dim
    query_states = self.q_proj(hidden_states)

    # batch_size, seq_len, num_heads_kv * head_dim
    key_states = self.k_proj(hidden_states)

    # batch_size, seq_len, num_heads_kv * head_dim
    value_states = self.v_proj(hidden_states)

    # batch_size, num_heads_q, seq_len, head_dim
    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # batch_size, num_heads_kv, seq_len, head_dim
    key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # batch_size, num_heads_kv, seq_len, head_dim
    value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # (batch_size, seq_len, head_dim), (batch_size, seq_len, head_dim)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

    # (batch_size, nunm_heads_q, seq_len, head_dim), (batch_size, num_heads_kv, seq_len, head_dim)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if kv_cache is not None:
      key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

    # repeat the keys and values to match num_heads_query
    key_states  = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Q * K_t / sqrt(head_dim)
    # batch_size, num_heads_q, seq_len_q, seq_len_kv
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    assert attention_mask is not None
    attn_weights = attn_weights + attention_mask

    # batch_size, num_heads_q, seq_len_q, seq_len_kv
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # batch_size, num_heads_q, seq_len_q, seq_len_kv * batch_size, num_heads_kv, seq_len_kv, head_dim --> batch_size, 
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
        f" {attn_output.size()}"
      )
    
    # transpose back to seq_len being the second dimension --> batch_size, num_heads_q, seq_len_q, head_dim --> batch_size, seq_len, num_heads_q, head_dim
    attn_output = attn_output.transpose(1, 2).contiguous()

    # concatenate all heads together --> batch_size, seq_len_q, num_heads_q, head_dim --> batch_size, seq_len_q, num_heads_q * head_dim
    attn_output = attn_output.view(batch_size, q_len, -1)

    # batch_size, seq_len_q, hidden_size
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
  def __init__(self, config, layer_idx):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
    self.mlp = GemmaMLP(config)
    self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def forward(self,
              hidden_states,
              attention_mask=None,
              position_ids=None,
              kv_cache=None):
    
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # b, seq_len, hidden_size
    hidden_states, _ = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      kv_cache=kv_cache
    )

    hidden_states = residual + hidden_states

    # b, seq_len, hidden_size
    residual = hidden_states
    
    # b ,seq_len, hidden_size
    hidden_states = self.post_attention_layernorm(hidden_states)

    # b ,seq_len, hidden_size
    hidden_states = self.mlp(hidden_states)

    # b ,seq_len, hidden_size
    hidden_states = residual + hidden_states

    return hidden_states


class GemmaModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
      [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )
    self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def get_input_embeddings(self):
    return self.embed_tokens

  def forward(self, attention_mask=None, position_ids=None, inputs_embeds=None, kv_cache=None):
    hidden_states=inputs_embeds
    normalizer=torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states=hidden_states*normalizer
    for decoder_layer in self.layers:

      # b, seq_len, hidden_size
      hidden_states = decoder_layer(
        # b, seq_len, hidden_size
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_cache=kv_cache
      )

    # b, seq_len, hidden_size
    hidden_states = self.norm(hidden_states)

    # b, seq_len, hidden_size
    return hidden_states


class GemmaForCausalLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = GemmaModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

  def get_input_embeddings(self):
    return self.model.embed_tokens
  
  def tie_weights(self):
    self.lm_head.weight = self.model.embed_tokens.weight

  def forward(self,
              attention_mask=None,
              position_ids=None,
              inputs_embeds=None,
              kv_cache=None):
    # inputs_embeds: b, seq_len, hidden_size
    # outputs: b, seq_len, hidden_size

    outputs = self.model(
      attention_mask=attention_mask,
      position_ids=position_ids,
      inputs_embeds=inputs_embeds,
      kv_cache=kv_cache
    )

    hidden_states = outputs
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    return_data = {
      "logits": logits,
    }

    if kv_cache is not None:
      # Return the updated cache
      return_data['kv_cache'] = kv_cache
    
    return return_data

class PaliGemmaMultiModalProjector(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

  def forward(self, image_features):
    hidden_states = self.linear(image_features)
    return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config
    self.vision_tower = SigLipVisionModel(config.vision_config)
    self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
    self.vocab_size = config.vocab_size

    language_model = GemmaForCausalLM(config.text_config)
    self.language_model = language_model

    self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
  
  # Follows a technique called weight tying
  def tie_weights(self):
    return self.language_model.tie_weights()
  
  def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, kv_cache=None):
    _, _, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

    # Combine the embeddings of the image tokens, texzt tokens, and mask out all the padding tokens
    final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    # b, seq_len: True for text tokens
    text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

    # b, seq_le: True for image tokens
    image_mask = input_ids == self.config.image_token_index

    # b, seq_len: True for padding tokens (for this, it will be false)
    pad_mask = input_ids == self.pad_token_id

    # Expanding the masks to accomodate for embedding dimension, needed to use torch.where
    text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

    # Cannot use torch.where here because the seq_length of scaled_image_features is not equal to seq_len of final_embeddings
    final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

    # Zero out paddig tokens
    final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    min_dtype = torch.finfo(dtype).min
    q_len = inputs_embeds.shape[1]

    if kv_cache == None or kv_cache.num_items() == 0:
      # No token masking
      # Only works when there is no padding
      causal_mask = torch.full(
        (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
      )

    else:
      # The query must be one single token
      assert q_len == 1
      kv_len = kv_cache.num_items() + q_len

      # We do not mask anything since each query needs to attend all previous queries, only works with no padding
      causal_mask = torch.full(
        (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
      )

    # Add the head dimension
    # b, q_len, kv_len --> b, num_heads, q_len, kv_len
    causal_mask = causal_mask.unsqueeze(1)

    if kv_cache is not None and kv_cache.num_items() > 0:
      # The position of the query is the last position
      position_ids = attention_mask.cumsum(-1)[:, -1]
      if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    else:
      #Create a position_id based on the size of attention mask, and for masked token, we can use number 1 as position
      position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

    return final_embedding, causal_mask, position_ids
  
  def forward(self, input_ids=None, pixel_values=None, attention_mask=None, kv_cache=None):
    assert torch.all(attention_mask == 1), "The input cannot be padded"

    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    # b, c, h, w --> b, num_patches, embed_dim
    selected_image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

    # b, num_patches, embed_dim --> b, num_patches, hidden_size
    image_features = self.multi_modal_projector(selected_image_features)

    # Merge embeddings of image and text tokens
    inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

    outputs = self.language_model(
      attention_mask=attention_mask,
      position_ids=position_ids,
      inputs_embeds=inputs_embeds,
      kv_cache=kv_cache
    )

    return outputs