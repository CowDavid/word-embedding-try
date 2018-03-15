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
def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')

    parser.add_argument('-tr', '--train', help='Train the model with corpus')

    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')

    parser.add_argument('-it', '--iteration', type=int, default=100000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=5000, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    
    parser.add_argument('-hi', '--hidden', type=int, default=50, help='size of word vector')
    
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    
    parser.add_argument('-co', '--context_size', type=int, default=2, help='The (n-1) of the n-gram')

    args = parser.parse_args()
    return args





class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs, embeds

def train_word_vector(corpus, n_iteration, hidden_size, context_size, learning_rate):
	#corpus_name = corpus.split('/')[-1].split('.')[0]
	voc, pairs = loadPrepareData(corpus)
	#print(corpus_name)
    # training data
	#print(pairs[1])
	CONTEXT_SIZE = context_size
	EMBEDDING_DIM = hidden_size
	test_sentence = []
	for i in range(len(pairs)):
		test_sentence.append(pairs[i][0].split())
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

	for epoch in range(n_iteration):
	    total_loss = torch.Tensor([0])
	    for context, target in trigrams:
	    	#print("context: {}".format(context))
	    	#print("target: {}".format(target))
	    	
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
	print(losses)  # The loss decreased every iteration over the training data!
	corpus_name = os.path.split(corpus)[-1].split('.')[0]
	directory = os.path.join(save_dir, 'model', corpus_name, '{}'.format(hidden_size))
	torch.save({
                'iteration': iteration,
                'w2v': model.state_dict(),
                'w2v_opt': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'backup_w2v_model')))

def test_word_vector(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
	checkpoint = torch.load(modelFile)
	voc, pairs = loadPrepareData(corpus)
	model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
	model.load_state_dict(checkpoint['w2v'])
	model.train(False);
	while(1):
		test_word = input('>')
		if test_word == 'q': break
		else: 
			get_word_vector(test_word, modelFile)
def get_word_vector(model, test_word):
	test_word_idxs = [voc.word2index[test_word]]
	test_word_var = Variable(torch.LongTensor(test_word_idxs))
	log_probs, embeds = model(test_word_var)
	print(embeds.data)



def run(args):
	reverse, fil, n_iteration, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input
	context_size = args.context_size
	if args.train:
	    train_word_vector(args.train, n_iteration, hidden_size, context_size, learning_rate)
	elif args.test:
		test_word_vector(args.test, args.corpus, hidden_size, context_size)
    


if __name__ == '__main__':
    args = parse()
    run(args)


