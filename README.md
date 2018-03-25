# word-embedding-try
訓練model:
	python try.py -tr ./data/movie_subtitles_en.txt -it 5000 -hi 300
輸入index，印出frequency:
	python try.py -te ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
畫出t-SNE降維後，所有字之間的二維分布(省去頻率500以下的字(frequency_boundary))
	python try.py -d ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
印出最接近man - (king - queen)的word
	python try.py -ter ./save/model/movie_subtitles_en/300/5000_backup_w2v_model.tar -c ./data/movie_subtitles_en.txt -hi 300
