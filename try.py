import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

from load import readVocs, loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from train import batch2TrainData
import argparse
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from tqdm import tqdm
import random
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
import time
def parse():
	parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
	parser.add_argument('-tr', '--train', help='Train the model with corpus')
	parser.add_argument('-te', '--test', help='Test the saved w2v model')
	parser.add_argument('-ter', '--test_vector_relation', help='Test the saved w2v model')
	parser.add_argument('-l', '--load', help='Load the model and train')
	parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
	parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
	parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
	parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
	parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
	parser.add_argument('-p', '--print', type=int, default=5000, help='Print every p iterations')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
	parser.add_argument('-hi', '--hidden', type=int, default=50, help='size of word vectors')
	parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
	parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
	parser.add_argument('-co', '--context_size', type=int, default=2, help='The (n-1) of the n-gram')
	parser.add_argument('-d', '--draw', help='Draw 2D word vector with the word vector model')
	args = parser.parse_args()
	return args

class NGramLanguageModeler(nn.Module):

	def __init__(self, vocab_size, embedding_dim, context_size):
		self.context_size = context_size
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size
		super(NGramLanguageModeler, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		embeds = torch.sum(embeds, 0)
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=0).view(1,self.vocab_size)
		return log_probs, embeds

def train_word_vector(corpus, n_iteration, hidden_size, context_size, learning_rate, batch_size):
	#corpus_name = corpus.split('/')[-1].split('.')[0]
	voc, pairs = loadPrepareData(corpus)
	#print(corpus_name)
    # training data
	#print(pairs[1])
	CONTEXT_SIZE = context_size
	EMBEDDING_DIM = hidden_size
	test_sentence = []
	for i in range(batch_size):
		pair = random.choice(pairs)
		test_sentence.append(pair[0].split())
		test_sentence[i].insert(0,"SOS")
		test_sentence[i].append("EOS")
	#print(test_sentence[:3])
	trigrams = []
	for j in range(len(test_sentence)):
		for i in range(len(test_sentence[j]) - 2):
			trigram = ([test_sentence[j][i], test_sentence[j][i + 1]], test_sentence[j][i + 2])
			trigrams.append(trigram)
	#print the first 3, just so you can see what they look like
	#print(trigrams[:30])
	#print(voc.n_words())
	#vocab = set(test_sentence)
	#word_to_ix = {word: i for i, word in enumerate(vocab)}
	losses = []
	loss_function = nn.NLLLoss()
	model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	it_count = 0
	tri_count = 0
	print("There are {} trigrams.".format(len(trigrams)))
	print("Total {} iterations.".format(n_iteration))
	for epoch in tqdm(range(n_iteration)):
		total_loss = torch.Tensor([0])
		it_count += 1
		tri_count = 0
		#if(it_count % 100 == 0):
			#print('\n')
			#print("Progressing iteration {}...".format(it_count))
		for context, target in trigrams:
			tri_count += 1
			#if(tri_count % 100 == 0):
			#	print("{} trigrams fed in this iteration...".format(tri_count))
			# Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
			# into integer indices and wrap them in variables)
			context_idxs = [voc.word2index[w] for w in context]
			context_var = Variable(torch.LongTensor(context_idxs))
			# Step 2. Recall that torch *accumulates* gradients. Before passing in a
			# new instance, you need to zero out the gradients from the old
			# instance
			model.zero_grad()
			# Step 3. Run the forward pass, getting log probabilities over next
			# words
			log_probs, embeds = model(context_var)
			# Step 4. Compute your loss function. (Again, Torch wants the target
			# word wrapped in a variable)
			loss = loss_function(log_probs, Variable(
			    torch.LongTensor([voc.word2index[target]])))
			# Step 5. Do the backward pass and update the gradient
			loss.backward()
			optimizer.step()
			total_loss += loss.data
		losses.append(total_loss)
	print('\n')
	print("Training completed!")
	print('\n')
	print("Loss: {}".format(losses))  # The loss decreased every iteration over the training data!
	corpus_name = os.path.split(corpus)[-1].split('.')[0]
	directory = os.path.join(save_dir, 'model', corpus_name, '{}'.format(hidden_size))
	if not os.path.exists(directory):
		os.makedirs(directory)
	torch.save({
				'iteration': n_iteration,
				'w2v': model.state_dict(),
				'w2v_opt': optimizer.state_dict(),
				'loss': loss
			}, os.path.join(directory, '{}_{}.tar'.format(n_iteration, 'backup_w2v_model')))

def test_word_vector(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
	checkpoint = torch.load(modelFile)
	voc, pairs = loadPrepareData(corpus)
	model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
	model.load_state_dict(checkpoint['w2v'])
	model.train(False)
	while(1):
		test_word = input('>')
		if test_word == 'q': break
		else:
			embeds = get_word_vector(model, voc.index2word[int(test_word)], voc, EMBEDDING_DIM)
			print("Word freauency of '{}': {}".format(voc.index2word[int(test_word)], \
				voc.word2count[voc.index2word[int(test_word)]]))
			#embeds = get_word_vector(model, test_word, voc, EMBEDDING_DIM)
			#print("The word vector of '{}': {}".format(test_word, embeds.data.view(1, EMBEDDING_DIM)))
def test_vector_relation(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
	checkpoint = torch.load(modelFile)
	voc, pairs = loadPrepareData(corpus)
	model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
	model.load_state_dict(checkpoint['w2v'])
	model.train(False)
	word1, word2, word3, word4 = "king", "queen", "man", "woman"
	test_word1 = np.array(get_word_vector(model, word1, voc, EMBEDDING_DIM).data)
	test_word2 = np.array(get_word_vector(model, word2, voc, EMBEDDING_DIM).data)
	test_word3 = np.array(get_word_vector(model, word3, voc, EMBEDDING_DIM).data)
	#test_word4 = np.array(get_word_vector(model, word4, voc, EMBEDDING_DIM).data)
	test_word4_like = test_word3 - (test_word1 - test_word2)
	#print(word1, ":\n",test_word1)#((test_word1 - test_word4_like) ** 2).mean(axis=None))
	#print(word2, ":\n",test_word2)#((test_word2 - test_word4_like) ** 2).mean(axis=None))
	#print(word3, ":\n",test_word3)#((test_word3 - test_word4_like) ** 2).mean(axis=None))
	#print(word4, ":\n",test_word4)#((test_word4 - test_word4_like) ** 2).mean(axis=None))
	#initial most_like
	#i_vector = np.array(get_word_vector(model, word1, voc, EMBEDDING_DIM).data)#most distant vector
	#initial_distance = ((i_vector - test_word4_like) ** 2).mean(axis=None)
	#print("initial_distance: ", initial_distance)
	
	_1st, _2nd, _3rd, _4th = 99999999, 99999999, 99999999, 99999999
	i_1st, i_2nd, i_3rd, i_4th = -1, -1, -1, -1
	for i in tqdm(range(0, voc.n_words)):
		i_vector = np.array(get_word_vector(model, voc.index2word[i], voc, EMBEDDING_DIM).data)
		distance = ((i_vector - test_word4_like) ** 2).mean(axis=None)
		#print(distance)
		if distance < _1st:
			_4th, _3rd, _2nd, _1st = _3rd, _2nd, _1st, distance
			i_4th, i_3rd, i_2nd, i_1st = i_3rd, i_2nd, i_1st, i
			#print("1st index:", i)
		elif distance < _2nd:
			_4th, _3rd, _2nd = _3rd, _2nd, distance
			i_4th, i_3rd, i_2nd = i_3rd, i_2nd, i
		elif distance < _3rd:
			_4th, _3rd = _3rd, distance
			i_4th, i_3rd = i_3rd, i
		elif distance < _4th:
			_4th = distance
			i_4th = i
	_1st_word = voc.index2word[i_1st]
	_2nd_word = voc.index2word[i_2nd]
	_3rd_word = voc.index2word[i_3rd]
	_4th_word = voc.index2word[i_4th]
	print("The most likely word of {} - ({} - {}) is {}".format(
		word3, word1, word2, _1st_word))

	print("Most likely words: {} > {} > {} > {} > other_words".format(_1st_word,
		_2nd_word, _3rd_word, _4th_word))
	
	'''
	vectors2D = np.concatenate(([test_word1], [test_word2], [test_word3], [test_word4]), axis = 0)
	
	time_start = time.time()
	tsne = TSNE()
	tsne_results = tsne.fit_transform(vectors2D)
	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
	tsne_results = tsne_results.reshape(2,vectors2D.shape[0])
	colors = ['b', 'c', 'y', 'm', 'r']
	king = plt.scatter(tsne_results[0][0],tsne_results[1][0], marker="x", color=colors[0])
	queen = plt.scatter(tsne_results[0][1],tsne_results[1][1], marker="o", color=colors[0])
	man = plt.scatter(tsne_results[0][2],tsne_results[1][2], marker="x", color=colors[1])
	woman = plt.scatter(tsne_results[0][3],tsne_results[1][3], marker="o", color=colors[1])
	plt.legend((king, queen, man, woman),
		('king', 'queen', 'man', 'woman'),
		scatterpoints=1,
        loc='lower right')
	corpus_name = os.path.split(corpus)[-1].split('.')[0]
	directory = os.path.join(save_dir, 'w2v_image', corpus_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	directory = os.path.join(directory,'relation_vectors2D.png')
	plt.savefig(directory, format='png')
	'''

def get_word_vector(model, test_word, voc, EMBEDDING_DIM):
	try:
		test_word_idxs = [voc.word2index[test_word]]
		test_word_var = Variable(torch.LongTensor(test_word_idxs))
		log_probs, embeds = model(test_word_var)
		return embeds
	except KeyError:
		print("Incorrect spelling.")
	
def draw_2D_word_vector(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
	checkpoint = torch.load(modelFile)
	voc, pairs = loadPrepareData(corpus)
	model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
	model.load_state_dict(checkpoint['w2v'])
	model.train(False);
	new_word = voc.index2word[0]
	new_word = np.array(get_word_vector(model, new_word, voc, EMBEDDING_DIM).data)
	vectors2D = np.array([new_word])
	start_word = 0
	index2vector = {start_word:new_word}
	nb_words = voc.n_words
	below1000_count = 0
	frequency_boundary = 500
	for i in range(start_word + 1, start_word + nb_words):
		new_word = voc.index2word[i]
		if voc.word2count[new_word] <= frequency_boundary:
			below1000_count += 1
		else:
			new_word = np.array(get_word_vector(model, new_word, voc, EMBEDDING_DIM).data)
			vectors2D = np.concatenate((vectors2D, [new_word]), axis = 0)
		#index2vector[i] = [new_word]
	print("{} words out of {} words are in low frequency({} times).".format(\
		below1000_count, voc.n_words, frequency_boundary))
	print("Shape of vectors2D: {}".format(vectors2D.shape))

	n_sne = nb_words
	print("t-SNE processing...")
	time_start = time.time()
	tsne = TSNE()
	tsne_results = tsne.fit_transform(vectors2D)
	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
	corpus_name = os.path.split(corpus)[-1].split('.')[0]
	tsne_results = tsne_results.reshape(2,vectors2D.shape[0])
	plt.scatter(tsne_results[0],tsne_results[1], marker=".")
	directory = os.path.join(save_dir, 'w2v_image', corpus_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	directory = os.path.join(directory,'({}, {})b{}vectors2D.png'.format(start_word, \
		start_word + nb_words-1, frequency_boundary))
	plt.savefig(directory, format='png')
	


def run(args):
	reverse, fil, n_iteration, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input
	context_size = args.context_size
	if args.train:
		train_word_vector(args.train, n_iteration, hidden_size, context_size, learning_rate, batch_size)
	elif args.test:
		test_word_vector(args.test, args.corpus, hidden_size, context_size)
	elif args.draw:
		draw_2D_word_vector(args.draw, args.corpus, hidden_size, context_size)
	elif args.test_vector_relation:
		test_vector_relation(args.test_vector_relation, args.corpus, hidden_size, context_size)
    

if __name__ == '__main__':
    args = parse()
    run(args)
