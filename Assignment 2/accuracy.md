# Accuracy

|SNo|Representation| Classification | Accuracy 
|---|---|---|---|
|1|Binary bag of words|Naive Bayes|84.716%|
|2|Binary bag of words|Logistic Regression|85.284%|
|3|Binary bag of words|Support Vector Machine (SVM)|84.172%|
|4|Binary bag of words|Feed Forward Neural Network|87.388%|
|5|Normalized Term frequency (tf)|Naive Bayes|84.864%|
|6|Normalized Term frequency (tf)|Logistic Regression|87.216%|
|7|Normalized Term frequency (tf)|Support Vector Machine (SVM)|60.504%|
|8|Normalized Term frequency (tf)|Feed Forward Neural Network|87.308%|
|9|Tfidf representation|Naive Bayes|86.424%|
|10|Tfidf representation|Logistic Regression|88.948%|
|11|Tfidf representation|Support Vector Machine (SVM)|67.972%|
|12|Tfidf representation|Feed Forward Neural Network|87.904%|
|13|Average Word2Vec (without tfidf)|Logistic Regression|79.892%|
|14|Average Word2Vec (without tfidf)|Support Vector Machine (SVM)|68.604%|
|15|Average Word2Vec (without tfidf)|Feed Forward Neural Network|79.772%|
|16|Average Word2Vec (with tfidf)|Logistic Regression|79.268%|
|17|Average Word2Vec (with tfidf)|Support Vector Machine (SVM)|71.636%|
|18|Average Word2Vec (with tfidf)|Feed Forward Neural Network|79.26%|
|19|Average GloVe (without tfidf)|Logistic Regression|80.38%|
|20|Average GloVe (without tfidf)|Support Vector Machine (SVM)|78.46%|
|21|Average GloVe (without tfidf)|Feed Forward Neural Network|78.056%|
|22|Average GloVe (with tfidf)|Logistic Regression|78.728%|
|23|Average GloVe (with tfidf)|Support Vector Machine (SVM)|77.292%|
|24|Average GloVe (with tfidf)|Feed Forward Neural Network|78.132%|
|25|Average Gensim Word2Vec (without tfidf)|Logistic Regression|87.888%|
|26|Average Gensim Word2Vec (without tfidf)|Support Vector Machine (SVM)|87.672%|
|27|Average Gensim Word2Vec (without tfidf)|Feed Forward Neural Network|87.344%|
|28|Average Gensim Word2Vec (with tfidf)|Logistic Regression|87.152%|
|29|Average Gensim Word2Vec (with tfidf)|Support Vector Machine (SVM)|87.052%|
|30|Average Gensim Word2Vec (with tfidf)|Feed Forward Neural Network|86.948%|
|31|Averaged Sentence Vectors|Logistic Regression|69.264%|
|32|Averaged Sentence Vectors|Support Vector Machine (SVM)|56.472%|
|33|Averaged Sentence Vectors|Feed Forward Neural Network|70.348%|
|34|Paragraph Vector|Logistic Regression|84.88%|
|35|Paragraph Vector|Support Vector Machine (SVM)|85.32%|
|36|Paragraph Vector|Feed Forward Neural Network|84.736%|
|37|GloVe|LSTM|84.454%|
|38|GloVe|GRU|84.228%|
|39|Word2Vec|LSTM|86.476%|
|40|Word2Vec|GRU|85.516%|
|41|Gensim Word2Vec|LSTM|84.736%|
|42|Gensim Word2Vec|GRU|84.728%|
|43|None(Embedding)|LSTM|83.216%|
|44|None(Embedding)|GRU|86.404%|