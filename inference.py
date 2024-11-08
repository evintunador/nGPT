import torch
from tqdm import tqdm

def sampler(
    logits: torch.Tensor,  # (batch_size, input_len, vocab_size)
    temperature: float = 1.0,  # controls randomness. set to 1.0 in order to not use temperature
    min_p: float = 0.05,  # min-p sampling threshold https://arxiv.org/abs/2407.01082
    top_k: int = None,  # max number of top tokens considered. set to None in order to not use
    top_p: float = None,  # cumulative probability threshold. set to 1.0 or preferably None in order to not use top_p
):
    """Generate token predictions from logits."""
    vocab_len, device = logits.shape[-1], logits.device

    logits = logits[:, -1, :]  # (batch_size=1, vocab_size)

    logits.div_(temperature)
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # both are (batch_size, vocab_size)

    # Min-p sampling
    if min_p is not None:
        probs_max = probs_sort[:, 0].unsqueeze(-1)
        min_p_threshold = min_p * probs_max
        min_p_mask = probs_sort < min_p_threshold
        probs_sort = torch.where(min_p_mask, 0, probs_sort)
        
    # Top-p filtering (if specified)
    if top_p is not None:
        probs_sum = torch.cumsum(probs_sort, dim=-1) # (batch_size, vocab_size)
        top_ps_mask = (probs_sum - probs_sort) > top_p # 0's are top-p selections & 1's are to be excluded
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

    # Top-k filtering (if specified)
    if top_k is not None:
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) >= top_k 
            # shape (vocab_size) tensor that iterates up by 1's
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) # (batch_size, vocab_size)
        top_ks_mask = top_ks_mask >= top_k

    # combining top-p with top-k and using whichever gives us fewer tokens
    if top_k is not None:
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalize
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)  
    # restore original order
    probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))  
    # sample from distribution
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    return next_token_id

def generate(
    prompt, # a single string
    model,  # nn.module that should output (batch_size,seq_len,vocab_len) and a list of dictionaries containing kv cache tensors
    tokenizer, # tokenizer of choice
    temperature: float = 1.0, # values above 1 increase entropy of predicted words. values near zero decrease entropy
    min_p: float = 0.05, # min-p sampling threshold https://arxiv.org/abs/2407.01082
    top_k: int = None, # optionally prevents model from sampling tokens that don't fit within the list of top k most likely tokens
    top_p: float = None, # optionally prevents the model from sampling tokens that don't fit within the cumsum range
    max_gen_len: int = None # maximum length you want the model to generate
):
    """Generate text from a prompt using the model and sampling settings."""
    # Convert min_p=0 to None to disable it if 0 was passed in
    min_p = None if min_p == 0 else min_p

    tokens_list = tokenizer.encode(prompt)
    max_gen_len = min(max_gen_len, model.max_seq_len - len(tokens_list))
    tokens = torch.tensor(tokens_list, device=model.device).to(torch.long).unsqueeze(0) # (batch_size = 1, max_prompt_len)
    
    ### now the actual inference loop.  
    for i in tqdm(range(max_gen_len), unit='tokens', leave=False):
        # first iteration is unique bc it has the entire prompt not just one query vector, so part of it is outside the for loop
        with torch.no_grad():
            logits, _ = model(input_token_ids = tokens)
    
        # turn the logits into probabilities and sample from them to get our next tokens
        next_token = sampler(logits, temperature, min_p, top_k, top_p)
        tokens = torch.cat((tokens, next_token), dim=-1)
        
        # if the model has outputted the eos token, we're done
        if next_token.item() == tokenizer.eos_id: break
    
    decoded_sequence = tokenizer.decode(tokens.squeeze(0).tolist())
    return decoded_sequence