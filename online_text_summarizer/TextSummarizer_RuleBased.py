import urllib.request
from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest


def getWPostText(url: str) -> str:
    """
    Parses Washington Post page to return the article text.


    Parameters:
            url (str): a Washington Post page url

    Returns:
            text (str): Binary string of the sum of a and b
    """

    page = urllib.request.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(page,"lxml")
    # WPost article text appears under the tag "article"
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    return text


def summarizeTextRB(text: str, n: int) -> str:
    """
    Returns the sentences containing the most words with the greatest frequency in the doc.
    

    Parameters:
            text : text to summarize
            n : number of top sentences to use as summary 

    Returns:
            list of top n sentences describing the text
    """

    # tokenize sentence
    sents = sent_tokenize(text)
    
    # n does not exceed sentences
    assert n <= len(sents)
    # tokenize words & rm stopwords
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation) + ['’'])
    word_sent=[word for word in word_sent if word not in _stopwords]
    
    # calculate term frequency
    freq = FreqDist(word_sent)

    # rank sentences by total word freq scores
    ranking = defaultdict(int)
    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
             
    # return n highest ranking sentences    
    sents_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sents_idx)][0]


if __name__ == "__main__":
    articleURL = "https://www.washingtonpost.com/washington-post-live/2021/01/13/artificial-intelligence-health-care/"

    text = getWPostText(articleURL)
    summary = summarizeTextRB(text, 3)
    print("Article summary\n" + summary)