### Program implementing transformer from scratch in pytorch - version 2, inspired from @ LucidRains

## Features:
# 1. Two necessary requirements for this implementation to work: (a) pad_token = 0 (automatically masks values) (b) keys = values (typically true for attention by definition)
# 2. Modifications over version 1 (tut13_0_transformer_from_scratch.py):
# 2.1. using in-built nn.Layernorm instead of a custom class for Layernorm
# 2.2. applying layernorm at input (as prenorm) rather than at output (this requires a separate layernorm at the final output of encoder)
# 2.3. GeLU activations in the FF block instead of ReLU
# 2.4. Learnable positional embeddings
# 2.5. The final embeddings (text_emb + pos_emb) are normalized using nn.Layernorm (followed by dropout)

## Todos / Questions:
# 1. Linear layers for keys, queries and values matrices should not have bias terms
# 2. dimensions of posenc - is the first dimension of input always maxlen? how is the second dimension set by div_term?
# 3. whats the use / effect of dropout in positional embeddings ?
# 4. attention calculation: batch size versus seq_len ?
# 5. why is the return value of MultiHeadAttention in reference code = dot_prod(query, key); without any role of the value
# 6. layernorm - why the (-1, keepdim) ?
# 7. decoder_layer - why two separate masks: src_mask and tgt_mask ?
# 8. generator not used in EncoderDecoder.forward() - in the reference code
# 9. in inference_test, why is vocab=11 when input=[1:10] ?
# 10. wouldn't dropout in attn_weights distort the probability distribution (no longer the weights sum up to 1)
# 11. extend pad masking to batches
# 12. target masking versus CrossEntropyLoss calculation

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len]
    mask = mask.unsqueeze(0).expand(batch_size, max_seq_len, max_seq_len) # mask.shape = [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token).unsqueeze(1) # mask.shape: [batch_size, 1, max_seq_len]
    mask = mask.expand(batch_size, max_seq_len, max_seq_len) # mask.shape: [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# utility function to calculate train loss (cross entropy loss)
def calculate_loss(model, minibatch, pad_token, device):
    enc_input, dec_input, target = minibatch[:,0], minibatch[:,1], minibatch[:,2] # minibatch.shape: [batch_size, 3, max_seq_len]
    enc_input, dec_input, target = torch.LongTensor(enc_input).to(device), torch.LongTensor(dec_input).to(device), torch.tensor(target).to(device) # convert to tensors

    # pad masking
    # pad masking is applied only on the basis of keys (no need to refer queries - this works out because keys = values in attention)
    enc_enc_mask = pad_mask(enc_input, pad_token)
    dec_dec_mask = pad_mask(dec_input, pad_token)
    enc_dec_mask = enc_enc_mask # since keys = enc_input for cross attention

    # add subsequent masking for dec_input
    sub_mask = subsequent_mask(dec_input.shape).to(device)
    dec_dec_mask = torch.logical_and(dec_dec_mask, sub_mask)

    # forward prop through model - note that we do not pass enc_dec_mask as its equal to enc_enc_mask
    scores = model(enc_input, enc_enc_mask, dec_input, dec_dec_mask) # scores.shape: [batch_size, max_seq_len, tgt_vocab]
    scores = scores.transpose(1, 2) # scores.shape: [batch_size, tgt_vocab, max_seq_len]
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(scores, target)
    return loss

# class to obtain embeddings from vocab tokens; embed_dim = d_model
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model) # the embedding vector is multiplied by sqrt(d_model)

# class to add positional encoding to embeddings (note that this class implements positional encoding as a constant untrainable vector)
class PositionalEncoding_Fixed(nn.Module):
    def __init__(self, d_model, dropout, maxlen=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # calculate positional encoding and save them (register buffer for saving params in params_dict that are not to be updated during backprop)
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1) # pos.shape: [maxlen, 1]
        div_term = 10000.0 * torch.exp( torch.arange(0, d_model, 2) / d_model ) # div_term.shape: [d_model/2]
        pe[:, 0::2] = torch.sin(pos / div_term) # pe[:, 0::2].shape: [maxlen, d_model/2]
        pe[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pe", pe)
    # add positional encoding to the embedding - and freeze positional encoding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        x = x + self.pe[x.size(1), :].requires_grad_(False) # but there are no trainable weights in positional encoding anyway ?!
        return self.dropout(x) # whats the use / effect of dropout in positional embeddings ?

# class to add positional encoding to embeddings (note that this class implements learnable positional encoding)
class PositionalEncoding_Learnt(nn.Module):
    def __init__(self, d_model, dropout, device):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    # add positional encoding to the embedding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        batch_size, max_seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        final_emb = self.dropout( self.norm(x + pos_emb) )
        return final_emb # whats the use / effect of dropout in positional embeddings ?

# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.GELU()
    def forward(self, x):
        return self.w2(self.dropout( self.act_fn(self.w1(x)) ))

# class implementing multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.output = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.attn_weights = None # placeholder to store attention weights (used to visualize attention matrices)
        self.dropout = nn.Dropout(dropout)

    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        # key.shape: [batch_size, seq_len, d_model]; mask.shape: [batch_size, seq_len, seq_len]
        # project key, query, value and reshape into multiple heads
        batch_size = key.shape[0]
        # (batch_size, seq_len, d_model) -proj-> (batch_size, seq_len, proj_dim) -view-> (batch_size, seq_len, n_heads, d_k) -transpose-> (batch_size, n_heads, seq_len, d_k)
        proj_key = self.W_K(key).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_query = self.W_Q(query).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_value = self.W_V(value).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2)
        # expand mask for n_heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # mask.shape: [batch_size, n_heads, seq_len, seq_len]
        # calculate attention
        attn_multihead, attn_weights = self.scaled_dotprod_attn(proj_key, proj_query, proj_value, mask, self.dropout)
        attn_multihead = attn_multihead.transpose(1, 2) # attn_multihead.shape: [batch_size, seq_len, n_heads, d_v]
        attn_multihead = torch.flatten(attn_multihead, start_dim=-2, end_dim=-1) # attn_multihead.shape: [batch_size, seq_len, n_heads * d_v]
        attn_multihead = self.output(attn_multihead) # attn_multihead.shape: [batch_size, seq_len, d_model]
        self.attn_weights = attn_weights
        return attn_multihead

    # function to calculate scaled dot product attention for one head
    def scaled_dotprod_attn(self, key, query, value, mask=None, dropout=None): # key.shape: [batch_size, seq_len, proj_dim]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) # attn_scores.shape: [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_weights = attn_scores.softmax(dim=-1) # attn_weights.shape: [batch_size, n_heads, seq_len, seq_len]
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        attn_vector = torch.matmul(attn_weights, value) # attn_vector.shape: [batch_size, n_heads, seq_len, d_v]
        return attn_vector, attn_weights

# # class implementing Layer Normalization
# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.a = nn.Parameter(torch.ones(dim))
#         self.b = nn.Parameter(torch.zeros(dim))
#         self.eps = eps
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a * ( (x - mean) / (std + self.eps) ) + self.b

# class implementing residual + normalization connection - takes in any block and applies a normalization + residual connection
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return x + self.dropout(sublayer( self.norm(x) )) # note that we apply the norm first

# class implementing a single encoder layer
# each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask)) # x.shape: [batch_size, seq_len, d_model]
        x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
        return x

# class implementing the entire encoder block = stacked encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at encoder output
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# class implementing a single decoder layer
# each decoder layer has three blocks: 1. (self) (masked) multihead attention 2. (src) (unmasked) multihead attention  3. feed_forward; with sublayer connection around each
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 3) # one for self_attn block, second for src_attn block, third for feed_forward block
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        m = encoder_out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # first apply self_attn block
        x = self.sublayers[1](x, lambda x: self.src_attn(m, x, m, src_mask)) # src_attn: (key from encoder, query from decoder, value from encoder)
        x = self.sublayers[2](x, self.feed_forward)
        return x

# class implementing the entire decoder block = stacked decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at decoder output
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)

# class implemeting the entire transformer = encoder + decoder
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, f_src_emb, f_tgt_emb, tgt_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.f_src_emb = f_src_emb
        self.f_tgt_emb = f_tgt_emb
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def encode(self, src, src_mask):
        encoder_out = self.encoder(self.f_src_emb(src), src_mask)
        return encoder_out
    def decode(self, tgt, encoder_out, src_mask, tgt_mask):
        decoder_out = self.decoder(self.f_tgt_emb(tgt), encoder_out, src_mask, tgt_mask)
        return decoder_out
    def forward(self, src, src_mask, tgt, tgt_mask):
        encoder_out = self.encode(src, src_mask) # encoder_out.shape: [batch_size, seq_len, d_model]
        decoder_out = self.decode(tgt, encoder_out, src_mask, tgt_mask) # decoder_out.shape: [batch_size, seq_len, d_model]
        scores = self.projection(decoder_out) # scores.shape: [batch_size, seq_len, tgt_vocab]
        return scores
    # function used for test time generation - outputs score only for last element in seq
    def generate(self, decoder_out, pred_index): # decoder_out.shape: [batch_size, seq_len, d_model]
        score = self.projection(decoder_out[:, pred_index]) # score.shape: [batch_size, d_model]
        return score


# caller function to instantiate the transformer model, using the defined hyperparams as input
def make_model(src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    # posenc = PositionalEncoding_Fixed(d_model, dropout) # positional encoding block (fixed)
    posenc = PositionalEncoding_Learnt(d_model, dropout, device) # positional encoding block (learnt)
    f_src_emb = nn.Sequential(Embeddings(src_vocab_size, d_model), deepcopy(posenc)) # function to convert src tokens into src embeddings
    f_tgt_emb = nn.Sequential(Embeddings(tgt_vocab_size, d_model), deepcopy(posenc)) # function to convert tgt tokens into tgt embeddings
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    decoder_layer = DecoderLayer(deepcopy(attn), deepcopy(attn), deepcopy(ff), d_model, dropout) # single decoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    decoder = Decoder(decoder_layer, n_layers, d_model) # decoder = stacked decoder layers
    model = EncoderDecoder(encoder, decoder, f_src_emb, f_tgt_emb, tgt_vocab_size) # the transformer model: encoder-decoder pair combining all modules

    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# utility function implementing greedy decoding
def greedy_decoding(encoder_out, enc_enc_mask, dec_input, max_seq_len, pad_token):
    for i in range(max_seq_len - 1):
        dec_dec_mask = pad_mask(dec_input, pad_token)
        dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
        decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
        score = model.generate(decoder_out, i) # score.shape: [1, tgt_vocab_size]
        logprobs = F.log_softmax(score, dim=-1)
        _, next_word_token = torch.max(logprobs, dim=-1)
        if torch.cuda.is_available():
            next_word_token = torch.cuda.LongTensor(next_word_token)
        else:
            next_word_token = torch.LongTensor(next_word_token)
        # update dec_input
        dec_input[0][i+1] = next_word_token
    return dec_input


# utility function implementing recursive beam search
def recursive_beam_search(curr_beam, encoder_out, recursion_depth, max_seq_len, beam_width, pad_token):
    all_beam_candidates = []
    # termination case
    if recursion_depth == max_seq_len:
        return curr_beam

    for dec_input, logprob in curr_beam:
        dec_dec_mask = pad_mask(dec_input, pad_token)
        dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
        decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
        score = model.generate(decoder_out, recursion_depth-1) # score.shape: [1, tgt_vocab_size]
        new_logprobs = F.log_softmax(score, dim=-1).squeeze(0)

        # beam candidates: list of tuples [ updated_dec_input, updated_score ]
        for class_token, new_logprob in enumerate(new_logprobs):
            if torch.cuda.is_available():
                next_word_token = torch.cuda.LongTensor([class_token])
            else:
                next_word_token = torch.LongTensor([class_token])
            updated_dec_input = deepcopy(dec_input)
            updated_dec_input[0][recursion_depth] = next_word_token
            updated_logprob = ( (logprob * (recursion_depth-1)) + new_logprob ) / recursion_depth # length normalized logprob
            new_beam_candidate = [ updated_dec_input, updated_logprob ]
            all_beam_candidates.append(new_beam_candidate)

    # select topk candidates
    all_beam_candidates = list(sorted(all_beam_candidates, key=lambda x: x[1], reverse=True))
    new_beam = all_beam_candidates[:beam_width]
    # recursive
    final_beam = recursive_beam_search(new_beam, encoder_out, recursion_depth+1, max_seq_len, beam_width, pad_token)
    return final_beam


# utility function implementing beam search decoding
def beam_search_decoding(encoder_out, dec_input, max_seq_len, beam_width, pad_token):
    all_beam_candidates = []
    recursion_depth = 1

    dec_dec_mask = pad_mask(dec_input, pad_token)
    dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
    decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
    score = model.generate(decoder_out, recursion_depth-1) # score.shape: [1, tgt_vocab_size]
    logprobs = F.log_softmax(score, dim=-1).squeeze(0)

    # beam candidates: list of tuples [ updated_dec_input, updated_score ]
    for class_token, logprob in enumerate(logprobs):
        if torch.cuda.is_available():
            next_word_token = torch.cuda.LongTensor([class_token])
        else:
            next_word_token = torch.LongTensor([class_token])
        updated_dec_input = deepcopy(dec_input)
        updated_dec_input[0][recursion_depth] = next_word_token
        updated_logprob = logprob
        new_beam_candidate = [ updated_dec_input, updated_logprob ]
        all_beam_candidates.append(new_beam_candidate)

    # select topk candidates
    all_beam_candidates = list(sorted(all_beam_candidates, key=lambda x: x[1], reverse=True))
    initial_beam = all_beam_candidates[:beam_width]
    # recursive beam search
    final_beam = recursive_beam_search(initial_beam, encoder_out, recursion_depth+1, max_seq_len, beam_width, pad_token)
    # pick out the best class_token_list
    best_dec_input, best_logprob = final_beam[0]
    return best_dec_input




# main
if __name__ == '__main__':
    # hyperparams
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 15 # useful for padding and recursion depth in beam search based translation
    beam_width = 3 # used for beam search decoding
    batch_size = 8
    lr = 1e-4
    num_epochs = 1000
    random_seed = 10

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # given parallel corpus for MT (English to French)
    corpus = [["He is jumping on the beach .", "Il saute sur la plage ."],
              ["She is running on the road .", "Elle court sur la route ."],
              ["She is climbing a mountain .", "Elle escalade une montagne ."],
              ["He is driving a car .", "Il conduit une voiture ."],
              ["She is standing on the bed .", "Elle est debout sur le lit ."],
              ["She is jumping on the car .", "Elle saute sur la voiture ."],
              ["He is running on the mountain .", "Il court sur la montagne ."],
              ["He is climbing on the bed .", "Il grimpe sur le lit ."],
              ["She is driving on the road .", "Elle roule sur la route ."],
              ["He is standing on the beach .", "Il est debout sur la plage ."]]

    ### create vocabularies and dataset

    dataset = []
    src_vocab, tgt_vocab, tgt_idx_vocab = {}, {}, {}
    n_unique_src_words, n_unique_tgt_words = 0, 0

    # add pad token in both src_vocab and tgt_vocab
    pad_token = 0
    src_vocab['<P>'] = pad_token
    tgt_vocab['<P>'] = pad_token
    tgt_idx_vocab[pad_token] = '<P>'
    n_unique_src_words += 1
    n_unique_tgt_words += 1

    # add start and end tokens to tgt vocab (only)
    start_token = 1
    end_token = 2
    tgt_vocab['<S>'] = start_token
    tgt_vocab['<E>'] = end_token
    tgt_idx_vocab[start_token] = '<S>'
    tgt_idx_vocab[end_token] = '<E>'
    n_unique_tgt_words += 2

    # process corpus
    for sp in corpus:
        enc_input, dec_input, target = [], [], []
        dec_input.append(start_token) # append start token to dec_input
        src_seq, tgt_seq = sp
        src_words = src_seq.split(" ")
        for word in src_words:
            if word not in src_vocab.keys():
                src_vocab[word] = n_unique_src_words
                n_unique_src_words += 1
            enc_input.append(src_vocab[word])
        tgt_words = tgt_seq.split(" ")
        for word in tgt_words:
            if word not in tgt_vocab.keys():
                tgt_vocab[word] = n_unique_tgt_words
                tgt_idx_vocab[n_unique_tgt_words] = word
                n_unique_tgt_words += 1
            dec_input.append(tgt_vocab[word])
            target.append(tgt_vocab[word])
        # append end token to target
        target.append(end_token)
        dataset_item = [enc_input, dec_input, target]
        # fill padding
        for item in dataset_item:
            while len(item) < max_seq_len:
                item.append(pad_token)
        # append to dataset
        dataset.append(dataset_item)
    dataset = np.array(dataset)

    # instantiate model
    model = make_model(len(src_vocab.keys()), len(tgt_vocab.keys()), d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = dataset[idx]

        # calculate loss
        loss = calculate_loss(model, minibatch, pad_token, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 100 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # test
    test_input = ["He is jumping on the beach .",
                  "She is running on the road .",
                  "She is climbing a mountain .",
                  "He is driving a car .",
                  "She is standing on the bed .",
                  "He is jumping on the road .",
                  "He is running on the car .",
                  "She is climbing on the beach .",
                  "She is driving on the bed .",
                  "He is standing on the mountain ."]

    print('\nTranslating test sentences:')
    model.eval()
    for sen in test_input:
        # convert input words to tokens
        words = sen.split(" ")
        enc_input = [src_vocab[w] for w in words]
        # pad
        while len(enc_input) < max_seq_len:
            enc_input.append(pad_token)
        enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device)
        enc_enc_mask = pad_mask(enc_input, pad_token)
        encoder_out = model.encode(enc_input, enc_enc_mask) # encoder_out.shape: [1, max_seq_len, d_model]

        ### decoding
        dec_input = [start_token]
        # pad
        while len(dec_input) < max_seq_len:
            dec_input.append(pad_token)
        dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device)

        # greedy decoding
        # dec_input = greedy_decoding(encoder_out, enc_enc_mask, dec_input, max_seq_len, pad_token)

        # beam search decoding
        dec_input = beam_search_decoding(encoder_out, dec_input, max_seq_len, beam_width, pad_token)

        # convert translated tokens to words
        pred_words = [tgt_idx_vocab[i.item()] for i in dec_input[0]]
        print('{} -> {}'.format(words, pred_words))
