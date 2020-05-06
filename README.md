# MisogynyClustering

Text and Multimedia Mining research project.

Structure:

`data` folder contains AMI training, testing and updated testing data sets

`related papers` folder contains papers devoted to hate speech in general, misogyny specifically and a couple of papers about NLP application to social networks content. `AMI` subforlder contains paper which explains the clallenge and a few reports of the participants.

`src` folder contains source code used for clustering `misogyny_clustering.py` and `utils.py` as well as auxilary files and the results of clustering analysis:
- `data` subfolder contains merged test and training set
- `img` - images with word clouds by category and elbow method results
- `results word occurences` - text files with 25 most frequent tokens by category excluding stop words
- `results` - clustering results. Each folder title explains the parameters used for clustering
- `{category}.txt` - auxilary files which contain tweets form a denoted category
