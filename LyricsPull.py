
import pandas as pd
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

# Data structures to store lyrics
lyrics_df = pd.DataFrame(columns=['artist', 'album', 'song', 'lyrics'])
artist_col = []
song_col = []
lyrics_col = []
album_col = []


def pull_lyrics(albumList, artist):
    for album in albumList:

        createFolder("Lyrics/" + artist + "/" + album)

        # Get all the nouns from each song
        for song in getSongList(artist, album):
            lyrics = getLyrics(song, artist)

            f = open("Lyrics/" + artist + "/" + album + "/" + song + ".txt", "w+")
            f.write(lyrics)
            # f = open("Lyrics/" + artist + "/" + album + "/" + song + ".txt", "r")
            # lyrics = f.read()

