Dataset:
The dataset is originally downloaded as a text document, for easier implementation we re-formatted the document and converted it into a csv. I’ve attached this altered data set in the directory titled “Petrov_Skevingotn_CodeAndDataset” since the code needs this to properly run. However, if you’re curious about the original dataset we’ve provided the download link below:


Original Download: https://cl.awaisathar.com/citation-sentiment-corpus/
Original Paper: https://aclanthology.org/P11-3015.pdf


Necessary Libraries:
* Tensorflow
* Pandas
* Matplotlib
* Os
* Numpy
* Re
* Nltk
* Keras
* Tqdm
* Sklearn


How to Run Code/Models:
1. Download the directory titled “Petrov_Skevington_CodeAndDataset”
2. Load the directory into your environment of choice (make sure that the GloVe embeddings and corpus are in the same directory as the .py files).
3. Run “LSTM_Model.py” to analyze the dataset using the LSTM model, the output will look like the LSTM IO.
4. Run “CNN+LSTM_Model.py” to analyze the dataset using the CNN model, the output will look like the CNN IO.