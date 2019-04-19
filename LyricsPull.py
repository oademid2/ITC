import scattertext as st
import pandas as pd
from pprint import pprint
import numpy as np
from scipy.stats import rankdata, hmean, norm
import spacy
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML
from LyricsMod import songLyrics, songGrab
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from scattertext import CorpusFromPandas, produce_scattertext_explorer

from lxml import html
import requests
import re
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from pymsgbox import *
import string



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# train text -- lowkey have no idea what this does rn  -- just know I need it for tagging as of now
train_text = songLyrics('slippery', 'migos').text()
sample_text = songLyrics('emotionless', 'Drake').text().lower()
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenize = custom_sent_tokenizer.tokenize(sample_text)


# First I need to pull the lyrics from the web -- I will use the package I created

# Function to get list of songs in album
def getSongList(artist, album_nameR):
    album_nameR = re.sub(r'\'', '-', album_nameR)  # remove apostrophes
    album_name = "-".join(word_tokenize(album_nameR))  # join with hyphen -- appropiate url format

    # scrape the page
    url = requests.get('https://genius.com/albums/' + artist + '/' + str(album_name) + '')  # url
    page = html.fromstring(url.content)
    scrape = page.xpath('//h3[@class = \'chart_row-content-title\']/text()')  # return content listed

    # Clean white space in entries
    for content, i in zip(scrape, range(len(scrape))):
        scrape[i] = content.strip()

    # remove empty entries & return
    return list(filter(lambda a: a != '',
                       scrape))  # list(filter(lambda a: len(songGrab().getLyrics(a, artist)) != 0, filter1))

# Function to get lyrics
def getLyrics(song, artist):
    song = re.sub("'", "", song)  # remove apostrophes
    song = re.sub(" ", "-", song)  # add dashes ---> making song title readable

    song_pg = requests.get('https://genius.com/' + artist + '-' + song + '-lyrics')  # url
    song_tree = html.fromstring(song_pg.content)
    song = song_tree.xpath('//div[@class = \'lyrics\']//text()')

    if len(song) == 0:
        # alert(text= song + ' not found', title='', button='OK')
        pass

    lyrics_holder = []

    for line in song:
        line = re.sub("[\(\[].*?[\)\]]", "", line)  # remove content in [*] ex/ [chorus: mariah carey]
        line = line.lstrip()  # remove white spaces
        lyrics_holder.append(line)  # add clean line into

    lines = list(filter(lambda a: a != '', lyrics_holder))  # filter out all the empty lines
    lyrics = ". ".join(lines)  # join all line as one paragraph

    return lyrics  # returns with each line as seperate element in list


# Albums I want to query
drake_albums_s1 = ['scorpion', 'views', 'more life', 'views', "If you\'re reading this it\'s too late"]
migos_albums_s1 = ['culture', 'migos']


def pull_lyrics(albumList, artist):
    for album in albumList:

        createFolder("Lyrics/" + artist + "/" + album)

        # Get all the nouns from each song
        for song in getSongList(artist, album):
            lyrics = getLyrics(song, artist)

            f = open("Lyrics/" + artist + "/" + album + "/" + song + ".txt", "w+")
            f.write(lyrics)
            print('++', lyrics)


# Data structures to store lyrics
lyrics_df = pd.DataFrame(columns=['artist', 'album', 'song', 'lyrics'])
lyricsdict = {}
artist_col = []
song_col = []
lyrics_col = []
album_col = []
nouns_col = []


#find all the nouns in the song

def localize_lyrics_nouns(albumList, artist):

    for album in albumList:

        # Get all the nouns from each song
        for song in getSongList(artist, album):

            f = open("Lyrics/" + artist + "/" + album + "/" + song + ".txt", "r")
            lyrics = f.read()

            nouns = []
            tokenized = custom_sent_tokenizer.tokenize(lyrics)

            # find nouns
            for word, i in zip(tokenized, range(len(tokenized))):
                words = nltk.word_tokenize(word)
                tagged = nltk.pos_tag(words)
                for i in range(len(words)):
                    if tagged[i][1] == 'NN':
                        # keep nouns
                        nouns.append(tagged[i][0])

            if lyrics != '':
                lyrics_col.append(lyrics)
                artist_col.append(artist)
                song_col.append(song)
                album_col.append(album)
                nouns_col.append(" ".join(nouns))


lyricsdict = {'artist': artist_col, 'album': album_col, 'song': song_col, 'lyrics': lyrics_col, 'nou': nouns_col}

#pull_lyrics(drake_albums_s1, 'drake')
#pull_lyrics(migos_albums_s1, 'migos')

localize_lyrics_nouns(drake_albums_s1, 'drake')
localize_lyrics_nouns(migos_albums_s1, 'migos')

df = pd.DataFrame(data=lyricsdict)
df.head()

#nlp = spacy.load('en')
nlp = st.whitespace_nlp_with_sentences
df['parsed'] = df.nou.apply(nlp)



# Scatter function modified from Jason Kesslet tutorial

def scatter():

    corpus = st.CorpusFromParsedDocuments(df, category_col='artist', parsed_col='parsed').build()
    term_freq_df = corpus.get_term_freq_df() #get term frequency
    print(term_freq_df)

    term_freq_df['drake_precision'] = term_freq_df['drake freq'] * \
                                    1./(term_freq_df['drake freq'] + term_freq_df['migos freq'])

    term_freq_df['drake_freq_pct'] = term_freq_df['drake freq'] * 1./term_freq_df['drake freq'].sum()
    term_freq_df['drake_hmean'] = term_freq_df.apply(lambda x: (hmean([x['drake_precision'], x['drake_freq_pct']])
                                                                       if x['drake_precision'] > 0 and x['drake_freq_pct'] > 0
                                                                       else 0), axis=1)
    term_freq_df.sort_values(by='drake_hmean', ascending=False).iloc[:10]
    print(term_freq_df.head())


    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())

    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())

    term_freq_df['drake_precision_normcdf'] = normcdf(term_freq_df['drake_precision'])
    term_freq_df['drake_freq_pct_normcdf'] = normcdf(term_freq_df['drake_freq_pct'])
    term_freq_df['drake_scaled_f_score'] = hmean(
        [term_freq_df['drake_precision_normcdf'], term_freq_df['drake_freq_pct_normcdf']])
    term_freq_df.sort_values(by='drake_scaled_f_score', ascending=False).iloc[:10]

    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['migos Score'] = corpus.get_scaled_f_scores('migos')
    term_freq_df['drake Score'] = corpus.get_scaled_f_scores('drake')
    print("Top 10 drake terms")
    pprint(list(term_freq_df.sort_values(by='drake Score', ascending=False).index[:10]))
    print("Top 10 migos terms")
    pprint(list(term_freq_df.sort_values(by='migos Score', ascending=False).index[:10]))

    html = produce_scattertext_explorer(corpus,
                                        category='drake',
                                        category_name='drake',
                                        not_category_name='migos',
                                        width_in_pixels=1000,
                                        minimum_term_frequency=5,
                                        transform=st.Scalers.scale,
                                        #metadata=lyrics_df['artist']
                                        )
    file_name = 'output/raw_freq.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)

    html = st.produce_scattertext_explorer(corpus,
                                           category='drake',
                                           category_name='drake',
                                           not_category_name='migos',
                                           minimum_term_frequency=5,
                                           width_in_pixels=1000,
                                           transform=st.Scalers.log_scale_standardize)
    file_name = 'output/logscale.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)

    html = produce_scattertext_explorer(corpus,
                                        category='drake',
                                        category_name='drake',
                                        not_category_name='migos',
                                        width_in_pixels=1000,
                                        minimum_term_frequency=5,
                                        transform=st.Scalers.percentile,
                                        #metadata=lyrics_df['speaker']
                                        )
    file_name = 'output/percentile.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)

    html = produce_scattertext_explorer(corpus,
                                        category='drake',
                                        category_name='drake',
                                        not_category_name='migos',
                                        width_in_pixels=1000,
                                        minimum_term_frequency=5,
                                        #metadata=convention_df['speaker'],
                                        term_significance=st.LogOddsRatioUninformativeDirichletPrior())
    file_name = 'output/alphabetic.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)

    def scale(ar):
        return (ar - ar.min()) / (ar.max() - ar.min())

    corner_scores = corpus.get_corner_scores('drake')
    frequencies_scaled = scale(np.log(term_freq_df.sum(axis=1).values))
    html = produce_scattertext_explorer(corpus,
                                        category='drake',
                                        category_name='drake',
                                        not_category_name='migos',
                                        minimum_term_frequency=5,
                                        width_in_pixels=1000,
                                        x_coords=frequencies_scaled,
                                        y_coords=corner_scores,
                                        scores=corner_scores,
                                        sort_by_dist=False,
                                        #metadata=lyrics_df['artist'],
                                        x_label='Log Frequency',
                                        y_label='Corner Scores')
    file_name = 'output/CornervsLog.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)

scatter()
