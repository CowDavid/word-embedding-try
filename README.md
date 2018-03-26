# word-embedding-try
#### Training:
Run this command to start training, change the argument values in your own need.
```
python try.py -tr ./data/movie_subtitles_en.txt -it 5000 -hi 300
```

Continue training with saved model:
```
python try.py -tr ./data/movie_subtitles_en.txt -l <MODEL_FILE_PATH> -it 5000 -hi 300
```
(optional arguments) -lr learning_rate -b batch_size(randomly extract some sentences with the number equals 'batch_size')
#### Testing
Input the index of a word, and then it prints out the times of the word appears in the corpus:
```
python try.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
```
#### Draw distribution graph
Draw the 2-dimensional distribution after doing dimension reduction using t-SNE(ignore the words in low frequency(frequency_boundary=500))
```
python try.py -d ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
```
####  man - (king - queen) = ?
Print out the closest word of 'man' - ('king' - 'queen')
```
python try.py -ter ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
```
