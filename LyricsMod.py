
#PACKAGE IMPORTS
import itertools

from lxml import html
import requests
import re
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from pymsgbox import *

#SCRAPE FOR ALBULM SONGS
class songGrab:

    # get list of songs
    def getSongList(self, artist, album_nameR):

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


    #get list of songs
    def getSongList(self, artist, album_nameR):

        album_nameR = re.sub(r'\'', '-', album_nameR)  # remove apostrophes
        album_name = "-".join(word_tokenize( album_nameR)) # join with hyphen -- appropiate url format

        # scrape the page
        url = requests.get('https://genius.com/albums/'+artist +'/'+str(album_name)+'')  #url
        page = html.fromstring(url.content)
        scrape = page.xpath('//h3[@class = \'chart_row-content-title\']/text()') # return content listed

        #Clean white space in entries
        for content,i in zip(scrape, range(len(scrape))):
                scrape[i] = content.strip()

        #remove empty entries & return
        return list(filter(lambda a: a != '', scrape))#list(filter(lambda a: len(songGrab().getLyrics(a, artist)) != 0, filter1))


    #get lyrics of song
    def getLyrics(self, song, artist):
        song =re.sub("'", "", song)  # remove content in [*] ex/ [chorus: mariah carey]
        song = re.sub(" ", "-", song)  # remove white spaces


        song_pg = requests.get('https://genius.com/'+artist + '-' + song + '-lyrics')  # url
        song_tree = html.fromstring(song_pg.content)
        songs = song_tree.xpath('//div[@class = \'lyrics\']//text()')
        if len(songs) == 0:
            #alert(text= song + ' not found', title='', button='OK')
            pass

        return songs #returns with each line as seperate element in list





class songLyrics:

    def __init__(self, song, artist):
        self.artist = artist
        self.song = song
        lyrics_holder = []

        for line in songGrab().getLyrics(song, artist):
            line = re.sub("[\(\[].*?[\)\]]", "", line)  # remove content in [*] ex/ [chorus: mariah carey]
            line = line.lstrip()  # remove white spaces
            lyrics_holder.append(line) #add clean line into

        lines = list(filter(lambda a: a != '', lyrics_holder)) #filter out all the empty lines
        self.lyrics = ". ".join(lines) # join all line as one paragraph


    def text(self):
        return self.lyrics

    def tokens(self):
        return TweetTokenizer().tokenize(self.lyrics);


#print(songLyrics('emotionless','Drake').text())










#
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"I know another girl that's crying out for help")
# for token in doc:
#     if token.text == 'girl':
#         print(token.text, token.dep_, token.head.text, token.head.pos_,
#           [child for child in token.children])


# filter1 = list(filter(lambda a: a != '', scrape))