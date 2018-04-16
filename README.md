# word-embedding-try
### Source of the corpus
https://github.com/Marsan-Ma/chat_corpus
### Source code
The codes aside from w2v.py are from https://github.com/ywk991112/pytorch-chatbot
### Training
#### Word2Vector
Run this command to start training, change the argument values in your own need.
```
python w2v.py -tr <CORPUS_FILE_PATH> -it 2500 -hi 300 -b 128
```
> python w2v.py -tr ./data/movie_subtitles_en.txt -it 2500 -hi 300 -b 128

Continue training with saved model.
```
python w2v.py -tr <CORPUS_FILE_PATH> -l <MODEL_FILE_PATH> -it 5000 -hi 300
```
> python w2v.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300

(optional arguments) -lr learning_rate -b batch_size(randomly extract some sentences with the number equals to 'batch_size')
#### Seq2seq
Train the seq2seq model with the pretrained word2vector model.
```
python main.py -tr <CORPUS_FILE_PATH> -pre <PRETRAINED_MODEL_FILE_PATH> -la 1 -hi 300 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```
> python main.py -tr ./data/movie_subtitles_en.txt -pre ./save/model/movie_subtitles_en/300/3000_backup_w2v_model.tar -la 1 -hi 300 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
### Test
Test the seq2seq model
#### test randomly
```
python main.py -te <SEQ2SEQ_MODEL_FILE_PATH> -pre .<PRETRAINED_MODEL_FILE_PATH> -c <CORPUS_FILE_PATH>
```
> python main.py -te ./save/model/movie_subtitles_en/1-1_300/26000_backup_bidir_model.tar -pre ./save/model/movie_subtitles_en/300/3000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt

### Word Frequency
Input the index of a word, and then it prints out the times of the word appears in the corpus.
```
python w2v.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python w2v.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Draw Distribution Graph
Draw the 2-dimensional distribution after doing dimension reduction using t-SNE(ignore the words in low frequency.
```
python w2v.py -d <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300 -fb <FREQUENCY_BOUNDARY(default=1500)>
```
> python w2v.py -d ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300 -fb 1500
### Draw Graph Manually
Draw the 2-dimensional distribution after doing dimension reduction using t-SNE(manually input the words you want to see).
```
python w2v.py -dm <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300 
```
> python w2v.py -dm ./save/model/movie_subtitles_en/300/2500_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Most Likely Word
Print out the closest word of the word you input in order to check if the model good or not.
```
python w2v.py -ter <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python w2v.py -ter ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Loss Graph
Draw the loss graph.
```
python w2v.py -lo <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python w2v.py -lo ./save/model/movie_subtitles_en/300/2500_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Predict Word
Predict the next word with previous two words.
```
python w2v.py -pr <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
>  python w2v.py -pr ./save/model/movie_subtitles_en/300/2500_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
