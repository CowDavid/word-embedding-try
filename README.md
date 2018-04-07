# word-embedding-try
### Source of the corpus
https://github.com/Marsan-Ma/chat_corpus
### Source code
The codes aside from try.py are from https://github.com/ywk991112/pytorch-chatbot
### Training
Run this command to start training, change the argument values in your own need.
```
python try.py -tr <CORPUS_FILE_PATH> -it 5000 -hi 300
```
> python try.py -tr ./data/movie_subtitles_en.txt -it 5000 -hi 300

Continue training with saved model.
```
python try.py -tr <CORPUS_FILE_PATH> -l <MODEL_FILE_PATH> -it 5000 -hi 300
```
> python try.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300

(optional arguments) -lr learning_rate -b batch_size(randomly extract some sentences with the number equals to 'batch_size')
### Word Frequency
Input the index of a word, and then it prints out the times of the word appears in the corpus.
```
python try.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python try.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Draw Distribution Graph
Draw the 2-dimensional distribution after doing dimension reduction using t-SNE(ignore the words in low frequency(frequency_boundary=500)).
```
python try.py -d <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python try.py -d ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
###  Man - (King - Queen) = ?
Print out the closest word of 'man' - ('king' - 'queen') in order to check if the model good or not.
```
python try.py -ter <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python try.py -ter ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Loss Graph
Draw the loss graph.
```
python try.py -lo <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python try.py -lo ./save/model/movie_subtitles_en/300/2500_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
### Predict Word
Predict the next word with previous two words.
```
python try.py -pr <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
>  python try.py -pr ./save/model/movie_subtitles_en/300/2500_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
