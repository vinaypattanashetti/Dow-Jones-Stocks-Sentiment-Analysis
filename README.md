# **Dow Jones Stock Sentiment Analysis**

### Problem Statement:
Investors struggle to predict stock market trends due to the vast amount of news data. Develop a model to analyze news headlines and determine their sentiment, providing insights into how news impacts Dow Jones stock prices and aiding in more informed investment decisions.

### Objective:

The goal is to analyze the sentiment of Dow Jones stock market headlines and predict whether the stock price will go up or down.

### Installation:

```
! pip install pandas numpy matplotlib seaborn nltk scikit-learn textblob
```

**Wordcloud Installation**
```
! pip install wordcloud
```

Clone this repository:
```
git clone https://github.com/vinaypattanashetti/Dow-Jones-Stocks-Sentiment-Analysis.git
cd Dow-Jones-Stocks-Sentiment-Analysis
```
### Usage:

Run the Jupyter notebook or Python script to perform the analysis:
```
jupyter notebook Stock-Sentiment_Analysis.ipynb
```
### Load the dataset:
```
df=pd.read_csv("news_headlines.csv")
```
### Exploratory data analysis:
 Data visualization: 
```
# Visualizing the count of 'Label' column from the dataset
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.countplot(x='Label', data=df)
plt.xlabel('Stock Sentiments-> 0 price goes down/Same, 1 price goes up)')
plt.ylabel('Count')
plt.show()
```

- Output:
<img src="https://github.com/user-attachments/assets/75250116-8e27-4b62-8345-cab77cfef157" width="400" height="300">

### Data Preprocessing:
**WordCloud Visualization**

<img src="https://github.com/user-attachments/assets/26cb7012-339c-4e92-b16b-2f79ef38989c" width="400" height="300">

### Vectorization
```
# Creating the Bag of Words model
cv = CountVectorizer(max_features=10000, ngram_range=(2,2))
X_train = cv.fit_transform(train_corpus).toarray()
```

### Model building, training and Evaluation:
Best model
```
# Model fitting for multinominal naive bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
```
Prediction:
```
nb_classifier.predict(X_test)
```

**Predicted the Model with evaluation result: 83.86% accuracy**

### Final prediction output: 
- News: Fifty four large investors managing 1 trillion pounds ($1.41 trillion) in assets have launched a campaign to curb the use of antibiotics in the meat and poultry.
- Prediction: The stock price will remain the same or will go down.

### Conclusion

- The project addresses the challenge investors face in predicting stock market trends amidst an overwhelming influx of news data. 

- By using Natural Language Processing (NLP) to analyze the sentiment of news headlines, the project aims to provide actionable insights into how news impacts Dow Jones stock prices. 

- This approach not only enhances the ability to forecast market movements but also empowers investors to make more informed decisions. 

- Ultimately, the integration of sentiment analysis into investment strategies can lead to more accurate predictions of stock price fluctuations, thereby improving investment outcome.
