# Automatic Music Genre Classification
This project is based on scikit-learn. The accuracy of 5-Fold cross-validation is 74.2%. The classifier used is MLP with (110,110) neurons. 
The confusion matrix is shown as below:
![](https://raw.githubusercontent.com/c-gongye/music_genre_classification/master/graph/validation/cm.png)    
The best performing class is classical:
![](https://raw.githubusercontent.com/c-gongye/music_genre_classification/master/graph/validation/classical.png)  
The worst performing class is rock:
![](https://raw.githubusercontent.com/c-gongye/music_genre_classification/master/graph/validation/rock.png)  
## Prerequisite
- python 2.7
- matplotlib
- scikit-learn
- FFmpeg (demo.py only)
## Dataset
[GTZAN](https://marsyasweb.appspot.com/download/data_sets/)  
Although we are grateful that the author provides this dataset for free. There are some flaws in this dataset, see
[An analysis of the GTZAN music genre dataset](http://dl.acm.org/citation.cfm?id=2390851)
and
[The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use](https://arxiv.org/abs/1306.1461)
Do not use it if you can.
- 10 genres
- each genre has 100 soundtracks  
## Usage
### feature_extraction.py
Please use -h to see descriptions and options.
X.npy and y.npy are the results of feature_extraction.py.
X is the extracted features, y is the corresponding labels.
### validation.py
Please use -h to see descriptions and options.
model.out is the trained model.
### demo.py
Please use -h to see descriptions and options.
It uses X.npy and model.out to predict the genre of any given music.

