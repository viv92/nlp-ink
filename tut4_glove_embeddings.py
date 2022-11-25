### Program implementing GloVe (global vector embeddings) on a minimal scale / corpus.

## Features:
# 1. GloVe objective = squared loss between dot product of word embeddings pair (u.v) and their co-occurence log probability log softmax(u.v)
# 2. The objective is weighted by a thereshold function to clip over-weighting of very frequent word pairs. We use log (co-occurence count + 1) as a proxy for the threshold function.

## Todos / questions:
# 1. The GloVe paper uses a single set of embeddings. But I get nans when using a single set of embeddings. So I'm using a pair of embeddings for each word for now (similar to word2vec).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class glove(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.V_embedding = nn.Linear(1, embed_dim) # parameters for center word
        self.U_embedding = nn.Linear(1, embed_dim) # parameters for context word
        self.vocab_size = vocab_size

    # return log probability: log p( context_word | center_word)
    def forward(self, context_word_idx, center_word_idx):
        context_word = self.U_embedding(context_word_idx) # shape: [batch_size, embed_dim]
        center_word = self.V_embedding(center_word_idx) # shape: [batch_size, embed_dim]
        dot_prods = torch.diag(torch.matmul(context_word, center_word.T)) # shape: [batch_size]
        numerator = torch.exp(dot_prods)
        all_word_idx = torch.arange(self.vocab_size, dtype=torch.float) # shape: [vocab_size]
        all_words = self.U_embedding(all_word_idx.unsqueeze(1)) # shape: [vocab_size, embed_dim]
        denominator = torch.sum(torch.exp( torch.matmul(center_word, all_words.T) ), dim=1) # shape: [batch_size]
        log_probs = torch.log(numerator) - torch.log(denominator)
        return log_probs, dot_prods

    # returns word embedding
    def get_embedding(self, x):
        embedding = self.V_embedding(x)
        return embedding


# function to extract random minibatch
def get_minibatch(dataset, co_occurence_freq_dict, batch_size):
    N = len(dataset)
    indices = np.arange(N)
    np.random.shuffle(indices)
    batch_indices = indices[:batch_size]
    minibatch = dataset[batch_indices]

    # get co-occurence counts for the minibatch
    co_occurence_counts = []
    for word1_idx, word2_idx in minibatch:
        word_pair_key = str(word1_idx) + str(word2_idx)
        co_occurence_counts.append(co_occurence_freq_dict[word_pair_key])
    co_occurence_counts = torch.FloatTensor(co_occurence_counts)

    inputs, targets = torch.FloatTensor(minibatch[:,0]), torch.FloatTensor(minibatch[:,1])
    return inputs, targets, co_occurence_counts


# function to calculate negative log likelihood loss over minibatch
def calculate_loss(model, inputs, targets, co_occurence_counts):
    log_probs, dot_prods = model(inputs.unsqueeze(1), targets.unsqueeze(1)) # log_probs.shape: [batch_size]
    # MSE
    loss = torch.mean( torch.log(co_occurence_counts+1) * torch.pow(log_probs - dot_prods, 2) )
    return loss


# visualize learnt embeddings on a 2-d plane (dimension reduced using PCA)
def visualize_embeddings(word_dict, model, vocab_size, embed_dim):
    inputs = np.arange(vocab_size) # idx of all words in vocab
    inputs = torch.FloatTensor(inputs)
    embeddings = model.get_embedding(inputs.unsqueeze(1)) # shape: [vocab_size, embed_dim]
    # reduce embedding dimension to 2
    U,S,V = torch.pca_lowrank(embeddings) # note that this centers the embeddings matrix internally
    embeddings_reduced = torch.matmul(embeddings, V[:, :2]) # shape: [vocab_size, 2]
    # plot
    for word, idx in word_dict.items():
        x,y = embeddings_reduced[idx].data.numpy()
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()



# main
if __name__ == '__main__':

    # hyperparams
    window_size = 2
    embed_dim = 10
    batch_size = 32
    lr = 1e-3
    num_epochs = 5000
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # given corpus
    corpus = ["He likes fruits", "He hates vegetables", "She hates fruits", "She likes vegetables", "She likes brocoli", "She hates orange", "He likes apple", "He hates cabagge"]

    # create vocab and word dict
    corpus_string = " ".join(corpus)
    word_list = corpus_string.split(' ')
    vocab = list(set(word_list))
    vocab.sort() # to get a deterministic ordering of words
    vocab_size = len(vocab)
    word_dict = {w:i for i,w in enumerate(vocab)}

    # create skip gram dataset and co-occurence freq dict
    skip_gram_dataset = []
    co_occurence_freq_dict = {}
    for i in range(window_size, len(word_list)-window_size):
        for j in range(i-window_size, i+window_size+1):
            if j == i:
                continue
            target_word_vocab_index = word_dict[word_list[i]]
            context_word_vocab_index = word_dict[word_list[j]]
            skip_gram_dataset.append([context_word_vocab_index, target_word_vocab_index])
            # for co-occurence freq dict
            word_pair_key = str(context_word_vocab_index) + str(target_word_vocab_index)
            if word_pair_key in co_occurence_freq_dict.keys():
                co_occurence_freq_dict[word_pair_key] += 1
            else:
                co_occurence_freq_dict[word_pair_key] = 1
    skip_gram_dataset = np.array(skip_gram_dataset)


    # instantiate model
    model = glove(embed_dim, vocab_size)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in range(num_epochs):

        # get random minibatch
        inputs, targets, co_occurence_counts = get_minibatch(skip_gram_dataset, co_occurence_freq_dict, batch_size)

        # calculate loss
        loss = calculate_loss(model, inputs, targets, co_occurence_counts)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 1000 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # visualize learned embeddings on a 2d plot (dimension reduced)
    visualize_embeddings(word_dict, model, vocab_size, embed_dim)
