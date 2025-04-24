# BirdsArentReal
Social Network Analysis of the #BirdsArentReal Hashtag

This project explores the user activity surrounding the satirical movement "Birds Aren’t Real", which humorously claims that birds have been replaced with surveillance drones. While ironic in nature, the movement serves as a form of social commentary on misinformation and the influence of digital media on public perception.

The analysis uses a publicly available dataset shared by Gabriel Preda on Kaggle (link below). The script ptaszki.py contains the complete analysis code.

Key Features of the Project:
- Exploratory analysis of 1903 tweets and 1326 users
- Bot detection (based on unverified accounts)
- Extraction of mentions and construction of a mentions network graph
- Network analysis: density, transitivity, centrality, clustering
- Identification of key users and discussion drivers
- Network visualization (all users vs. only verified users)
- Topic clustering using BERTopic and WordCloud

Tools and Technologies:
Python (pandas, numpy, networkx, matplotlib, wordcloud, BERTopic, NLTK, TextBlob)

Dataset: https://www.kaggle.com/datasets/gpreda/birds-arent-real

This project demonstrates how data analysis and network modeling can be used to uncover the structure and dynamics of online social movements — even the most absurd ones.
