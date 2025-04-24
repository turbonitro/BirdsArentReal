import pandas as pd
import numpy as np
import re
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

twitter = pd.read_csv(r'birds.csv')

twitter.head()

user_tweet_counts = (
    twitter.groupby('user_name')
           .agg(user_description=('user_description', 'first'), tweet_count=('user_name', 'size'))
           .rename(columns={'user_name': 'tweet_count'})  #Zmiana nazwy kolumny liczby tweetów
           .reset_index()
           .sort_values(by='tweet_count', ascending=False)
)
print(user_tweet_counts)

#Deklaracja zmiennych
twitter['mentions'] = twitter['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
G = nx.DiGraph()
verified_users = twitter[twitter['user_verified'] == True]
bots = twitter[twitter['user_verified'] == False]
hashtag_data = twitter['hashtags'].dropna().str.cat(sep=' ')
unique_users = twitter[['user_name', 'user_description']].drop_duplicates()
unique_users.columns = ['name', 'description']
desc_data = unique_users['description'].dropna().str.cat(sep=' ')
mention_pairs = verified_users['mentions'].apply(lambda mentions: list(combinations(mentions, 2)))

mention_pairs = [pair for sublist in mention_pairs for pair in sublist]  # Flatten the list of pairs

#stworzenie dataframe dla wybranych par
mention_df = pd.DataFrame(mention_pairs, columns=['source', 'target'])
mention_df = mention_df.groupby(['source', 'target']).size().reset_index(name='weight')

#gfraf
G_verified = nx.from_pandas_edgelist(mention_df, 'source', 'target', create_using=nx.DiGraph())

#statystyki sieci
verified_network_density = nx.density(G_verified)
verified_network_transitivity = nx.transitivity(G_verified)
verified_network_loops = sum(1 for _ in nx.selfloop_edges(G_verified))
verified_network_average_clustering = nx.average_clustering(G_verified)
verified_network_degree_assortativity = nx.degree_assortativity_coefficient(G_verified)

print(f"Gęstość sieci (zweryfikowani): {verified_network_density}")
print(f"Przechodniość (zweryfikowani): {verified_network_transitivity}")
print(f"Liczba pętli (zweryfikowani): {verified_network_loops}")
print(f"Średni współczynnik klastra (zweryfikowani): {verified_network_average_clustering}")
print(f"Współczynnik korelacji stopnia (zweryfikowani): {verified_network_degree_assortativity}")

#wizualizacja
degree_dict_verified = dict(G_verified.degree())
node_colors_verified = ['violet' if degree >= 3 else 'darkviolet' for degree in degree_dict_verified.values()]

plt.figure(figsize=(10, 8))
pos = nx.kamada_kawai_layout(G_verified)  # or any other layout
nx.draw(G_verified, pos, node_color=node_colors_verified, with_labels=True, node_size=[v * 100 for v in degree_dict_verified.values()], edge_color='blue', font_color='black', font_size=8)
nx.draw_networkx_edge_labels(G_verified, pos, edge_labels=nx.get_edge_attributes(G_verified, 'weight'))
plt.title('Zweryfikowani użytkownicy wierzący, że ptaki nie są prawdziwe', fontsize=16)
plt.show()

mentions_users = []
for mentions in twitter['mentions']:
    mentions_users += list(combinations(mentions, 2))
user_callout = pd.DataFrame(mentions_users, columns=['actor_1', 'actor_2']).value_counts().reset_index()
user_callout.columns = ['actor_1', 'actor_2', 'call_count']

#użytkownicy z liczbą interakcji większą lub równą 3
graph_data_above2 = user_callout[user_callout['call_count']>=3]
print(graph_data_above2)

#Analiza par wzmianek na swój temat
for index, row in graph_data_above2.iterrows():
    G.add_edge(row['actor_1'], row['actor_2'], weight=row['call_count'])

#stopień każdego węzła
degree_dict = dict(G.degree())
node_colors = ['violet' if degree >= 5 else 'darkviolet' for degree in degree_dict.values()]

plt.figure(figsize=(10, 8))
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, node_size=[v * 800 for v in degree_dict.values()], font_size=8, font_color='black', edge_color='deeppink', arrows=True, node_color=node_colors)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Użytkownicy wierzący, że ptaki nie są prawdziwe', fontsize=20)
plt.show()

text_data = twitter['text'].dropna().str.cat(sep=' ')

text_data = text_data.lower()
text_data = re.sub(r'http\S+', '', text_data)
text_data = text_data.replace("'", '')
text_data = text_data.replace('birdsarentreal', '')
text_data = text_data.replace("birds", '')
text_data = text_data.replace('arentreal', '')
text_data = text_data.replace('arent real', '')
text_data = text_data.replace('t real', '')
text_data = text_data.replace('aren', '')

wordcloud = WordCloud(width=1600, height=800, colormap='magma').generate(text_data)

plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

verified_users.loc[:, 'mentions'] = verified_users['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
mentions_users = []
for mentions in verified_users['mentions']:
    mentions_users += list(combinations(mentions, 2))
verified_user_callout = pd.DataFrame(mentions_users, columns=['actor_1', 'actor_2']).value_counts().reset_index()
verified_user_callout.columns = ['actor_1', 'actor_2', 'call_count']
print(verified_user_callout)

for index, row in verified_user_callout.iterrows():
    G.add_edge(row['actor_1'], row['actor_2'], weight=row['call_count'])

#to samo tylko na userach zweryfikowanych
degree_dict = dict(G.degree())
node_colors = ['violet' if degree >= 3 else 'darkviolet' for degree in degree_dict.values()]

plt.figure(figsize=(10, 8))
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, node_size=[v * 400 for v in degree_dict.values()], font_size=8, font_color='black', edge_color='deeppink', arrows=True, node_color=node_colors)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Zweryfikowani użytkownicy wierzący, że ptaki nie są prawdziwe', fontsize=20)
plt.show()

verified_users.loc[:, 'mentions'] = verified_users['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
all_mentions = [mention for sublist in verified_users['mentions'] for mention in sublist]
all_usernames = [user_name for sublist in verified_users['user_name']for user_name in sublist]
text_data = verified_users['text'].dropna().str.cat(sep=' ')
text_data = text_data.lower()
for mention in all_mentions:
    text_data = text_data.replace('@' + mention, '')
for user_name in all_usernames:
    test_data = text_data.replace(user_name, '')
text_data = re.sub(r'http\S+', '', text_data)
text_data = text_data.replace("'", '')
text_data = text_data.replace('birdsarentreal', '')
text_data = text_data.replace("birds", '')
text_data = text_data.replace('arentreal', '')
text_data = text_data.replace('arent real', '')
text_data = text_data.replace('t real', '')
text_data = text_data.replace('aren', '')
wordcloud = WordCloud(width=1600, height=800, colormap='inferno').generate(text_data)

plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

bots.loc[:, 'mentions'] = bots['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
mentions_users = []
for mentions in bots['mentions']:
    mentions_users += list(combinations(mentions, 2))
bot_callout = pd.DataFrame(mentions_users, columns=['actor_1', 'actor_2']).value_counts().reset_index()
bot_callout.columns = ['bot_1', 'bot_2', 'call_count']

bot_data_above2 = bot_callout[bot_callout['call_count']>=3]

for index, row in bot_data_above2.iterrows():
    G.add_edge(row['bot_1'], row['bot_2'], weight=row['call_count'])

#węzły na botach
degree_dict = dict(G.degree())
node_colors = ['darkslategray' if degree >= 3 else 'teal' for degree in degree_dict.values()]

plt.figure(figsize=(10, 8))
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, node_size=[v * 400 for v in degree_dict.values()], font_size=8,
        font_color='black', edge_color='darkgreen', arrows=True, node_color=node_colors)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Boty twierdzące, że ptaki nie są prawdziwe', fontsize=20)
plt.show()

bots.loc[:, 'mentions'] = bots['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
all_mentions = [mention for sublist in bots['mentions'] for mention in sublist]
all_usernames = [user_name for sublist in bots['user_name']for user_name in sublist]
text_data = bots['text'].dropna().str.cat(sep=' ')
text_data = text_data.lower()
for mention in all_mentions:
    text_data = text_data.replace('@' + mention, '')
for user_name in all_usernames:
    test_data = text_data.replace(user_name, '')
text_data = re.sub(r'http\S+', '', text_data)
text_data = text_data.replace("'", '')
text_data = text_data.replace('birdsarentreal', '')
text_data = text_data.replace("birds", '')
text_data = text_data.replace('arentreal', '')
text_data = text_data.replace('arent real', '')
text_data = text_data.replace('t real', '')
text_data = text_data.replace('aren', '')

wordcloud = WordCloud(width=1600, height=800, colormap='ocean').generate(text_data)

plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

hashtag_data = hashtag_data.lower()
hashtag_data = hashtag_data.replace("'", '')
hashtag_data = hashtag_data.replace('birdsarentreal', '')
hashtag_data = hashtag_data.replace("birds", '')
hashtag_data = hashtag_data.replace('arentreal', '')
hashtag_data = hashtag_data.replace('arent real', '')
hashtag_data = hashtag_data.replace('t real', '')
hashtag_data = hashtag_data.replace('aren', '')
hashtag_data = hashtag_data.replace('https', '')

wordcloud = WordCloud(width=1600, height=800, colormap='nipy_spectral').generate(hashtag_data)

plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

desc_data  = desc_data.lower()
desc_data = re.sub(r'http\S+', '', desc_data)
wordcloud = WordCloud(width=1600, height=800, colormap='twilight_shifted').generate(desc_data)

plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(wordcloud, interpolation='spline36')
plt.axis('off')
plt.show()

# Centralność (degree centrality, betweenness centrality, closeness centrality, eigenvector centrality)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
closeness_centrality = nx.closeness_centrality(G)

try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
except nx.PowerIterationFailedConvergence:
    eigenvector_centrality = "Nie udało się obliczyć eigenvector centrality - nie zbieżność algorytmu."

print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)
print("Eigenvector Centrality:", eigenvector_centrality)

# Wykrywanie społeczności (community detection)
from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G))
print(f"Liczba wykrytych społeczności: {len(communities)}")

# Dodanie informacji o przynależności do społeczności do węzłów
node_community = {}
for i, community in enumerate(communities):
    for node in community:
        node_community[node] = i

nx.set_node_attributes(G, node_community, 'community')

# Wizualizacja sieci z informacją o przynależności do społeczności
node_colors = [node_community[node] for node in G.nodes()]

plt.figure(figsize=(10, 8))
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, node_size=400, node_color=node_colors, cmap=plt.cm.rainbow, font_size=8, font_color='black', edge_color='deeppink', arrows=True)
plt.title('Wizualizacja sieci z informacją o społecznościach', fontsize=20)
plt.show()

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = ' '.join(re.sub("(nan)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = word_tokenize(text)
    remove_sw = [word for word in text if not word in stopwords.words()]
    corpus = " ".join(remove_sw)
    return corpus

twitter['text'] = twitter['text'].astype(str).apply(clean_text)

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(twitter['text'].tolist())

topic_model.visualize_barchart()

#analiza sentyentu
#funkcja do obliczania sentymentu
def calculate_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None

twitter['sentiment'] = twitter['text'].apply(calculate_sentiment)

#dodanie klasyfikacji sentymentu jako pozytywny, neutralny, negatywny
twitter['sentiment_type'] = twitter['sentiment'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

sentiment_summary = twitter['sentiment_type'].value_counts()
print(sentiment_summary)
