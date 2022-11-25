### Program implementing word2vec (skip gram model) on a minimal scale / corpus.

## Features:
# 1. This implementation uses cross entropy loss for training the word embeddings. Using the CE loss imposes two requirements (not needed for negative log likelihood loss):
# 1.a. The target words be replaced by zero indexed class labels [0, C), where C = num_classes = vocab_size
# 1.b Dimension of the unnormalized score vector be equal to num_classes

## Todos / questions:
# 1.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class word2vec(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.W_embedding = nn.Linear(1, embed_dim)
        self.W_score = nn.Linear(embed_dim, vocab_size)

    # return raw (unnormalized score vector)
    def forward(self, x):
        embedding = self.W_embedding(x)
        raw_score_vector = self.W_score(embedding)
        return raw_score_vector

    # returns word embedding
    def get_embedding(self, x):
        embedding = self.W_embedding(x)
        return embedding


# function to extract random minibatch
def get_minibatch(dataset, batch_size):
    N = len(dataset)
    indices = np.arange(N)
    np.random.shuffle(indices)
    batch_indices = indices[:batch_size]
    minibatch = dataset[batch_indices]
    inputs, targets = torch.FloatTensor(minibatch[:,0]), torch.tensor(minibatch[:,1])
    return inputs, targets


# function to calculate cross entropy loss over minibatch
def calculate_loss(model, inputs, targets):
    scores = model(inputs.unsqueeze(1)) # scores.shape: [batch_size, num_classes=vocab_size]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, targets)
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

    # instantiate model
    model = word2vec(embed_dim, vocab_size)

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
