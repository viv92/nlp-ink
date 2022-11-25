### Program demonstrating BPE algorithm via an example
# The algo works as follows:
# 1. All the reference text is broken to smallest units (characters)
# 2. Initial vocab dict is a freq dict of smallest units (characters)
# 3. The pseudocode for rest of the algo is as follows:
#       curr_vocab = init_vocab
#       for i in range(max_iters):
#           curr_tokens = curr_vocab.keys()
#           pair_dict = {freq dict of all curr_token pairs occuring in the reference text}
#           best_pair = [curr_token_i, curr_token_j] - the curr_token pair with highest freq in pair_dict
#           new_token = ''.join(curr_token_i, curr_token_j)
#           new_token_freq = pair_dict[best_pair]
#           new_vocab = curr_vocab
#           new_vocab[new_token] = new_token_freq # add the new token with its freq
#           new_vocab[curr_token_i] -= new_token_freq # decrement the freq of the constituent token that formed the new token
#           new_vocab[curr_token_j] -= new_token_freq
#           curr_vocab = new_vocab # iterate

## Note: SentencePiece software employs the same BPE algorithm, but it applies it on Unicode(reference_text) rather than directly applying it on reference text.

import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print('best pair: ', best)
    print('new vocab: ', vocab)
