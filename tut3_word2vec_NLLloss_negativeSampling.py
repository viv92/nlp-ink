### Program implementing word2vec (skip gram model) on a minimal scale / corpus.

## Features:
# 1. This implementation uses negative log likelihood loss with negative sampling. However we don't use pytorch's inbuilt NLLLoss(), since we don't provide target separately. The target word is specified in the objective via dot product with the context word.
# 2. Two sets of embeddings for each word.

## Todos / questions:
# 1. Modify objective for negative sampling (sigmoid instead of softmax)
# 2. Vectorizing the negative sampling across batches for faster processing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class word2vec(nn.Module):
    def __init__(self, embed_dim, vocab_size, unigram_probs, num_neg_samples):
        super().__init__()
        self.V_embedding = nn.Linear(1, embed_dim) # parameters for center word
        self.U_embedding = nn.Linear(1, embed_dim) # parameters for context word
        self.vocab_size = vocab_size
        self.unigram_probs = unigram_probs
        self.num_neg_samples = num_neg_samples

    # return log probability: log p( context_word | center_word)
    def forward(self, context_word_idx, center_word_idx):
        context_word = self.U_embedding(context_word_idx) # shape: [batch_size, embed_dim]
        center_word = self.V_embedding(center_word_idx) # shape: [batch_size, embed_dim]
        numerator = F.sigmoid( torch.diag(torch.matmul(context_word, center_word.T)) ) # shape: [batch_size]

        # get negative samples - cannot think of a way of vectorizing this across batches. So going batchwise for now.
        batch_size = center_word.shape[0]
        log_probs = torch.empty(batch_size)

        for batch_idx in range(batch_size):
            center_word_id = center_word_idx[batch_idx]
            neg_words_idx = [center_word_id]
            while center_word_id in neg_words_idx:
                neg_words_idx = np.random.choice(self.vocab_size, size=self.num_neg_samples, replace=True)
            neg_words_idx = torch.FloatTensor(neg_words_idx)
            neg_words = self.U_embedding(neg_words_idx.unsqueeze(1)) # shape: [num_neg_samples, embed_dim]
            denominator = torch.sum(F.sigmoid( torch.matmul(center_word[batch_idx].unsqueeze(0), neg_words.T) )) # shape: [batch_size]
            log_probs[batch_idx] = torch.log(numerator[batch_idx]) - torch.log(denominator)

        return log_probs


    # returns word embedding
    def get_embedding(self, x):
        embedding = self.V_embedding(x)
        return embedding


# function to extract random minibatch
def get_minibatch(dataset, batch_size):
    N = len(dataset)
    indices = np.arange(N)
    np.random.shuffle(indices)
    batch_indices = indices[:batch_size]
    minibatch = dataset[batch_indices]
    inputs, targets = torch.FloatTensor(minibatch[:,0]), torch.FloatTensor(minibatch[:,1])
    return inputs, targets


# function to calculate negative log likelihood loss over minibatch
def calculate_loss(model, inputs, targets):
    log_probs = model(inputs.unsqueeze(1), targets.unsqueeze(1)) # log_probs.shape: [batch_size]
    loss = -torch.mean(log_probs)
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
    embed_dim = 5
    batch_size = 32
    lr = 1e-3
    num_epochs = 5000
    num_neg_samples = 10
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # given corpus
    corpus = ["He likes fruits", "He hates vegetables", "She hates fruits", "She likes vegetables",\
                "She likes brocoli", "She hates orange", "He likes apple", "He hates cabagge"]

    # create vocab and word dict
    corpus_string = " ".join(corpus)
    word_list = corpus_string.split(' ')
    vocab = list(set(word_list))
    vocab.sort() # to get a deterministic ordering of words
    vocab_size = len(vocab)
    word_dict = {w:i for i,w in enumerate(vocab)}

    # create skip gram dataset
    skip_gram_dataset = []
    for i in range(window_size, len(word_list)-window_size):
        for j in range(i-window_size, i+window_size+1):
            if j == i:
                continue
            target_word_vocab_index = word_dict[word_list[i]]
            context_word_vocab_index = word_dict[word_list[j]]
            skip_gram_dataset.append([context_word_vocab_index, target_word_vocab_index])
    skip_gram_dataset = np.array(skip_gram_dataset)

    # create unigram probs
    unigram_freq = np.zeros(vocab_size)
    for w in word_list:
        unigram_freq[word_dict[w]] += 1
    unigram_probs = unigram_freq ** (3/4)
    unigram_probs = unigram_probs / sum(unigram_probs)

    # instantiate model
    model = word2vec(embed_dim, vocab_size, unigram_probs, num_neg_samples)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in range(num_epochs):

        # get random minibatch
        inputs, targets = get_minibatch(skip_gram_dataset, batch_size)

        # calculate loss
        loss = calculate_loss(model, inputs, targets)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 1000 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # visualize learned embeddings on a 2d plot (dimension reduced)
    visualize_embeddings(word_dict, model, vocab_size, embed_dim)
