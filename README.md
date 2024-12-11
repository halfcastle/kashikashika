# kashikashika

# kashikashika

here's what the code does now:
- **beautifulsoup scraper**: crawls uta-net for all lyrics from a specific artist and saves them in a dataframe with song title, lyrics, and lyricist. formats the csv so that all lyricist names are consistent (e.g. 中島真一 = Nakajin)
- **janome**, the heart and soul of this whole project: a Japanese lexicon tokenizer. read the official documentation [here](https://mocobeta.github.io/janome/). I ran janome across my lyric files and extracted individual dataframes containing nouns, adjectives, and verbs. however, janome has its limitations when you use it on sekaowa lyrics, because it cannot parse anything written in English.
- after (quite) a bit of data cleaning, you can now use a simple **WordCloud** function to create the wordcloud visualizations according to word frequency.

<p align="center">
  <img src="/ksksk/images/nouns_cloud.png" width="800">
</p>

- **compares the nouns** against an [emotional polarity table](http://www.lr.pi.titech.ac.jp/~takamura/pndic_ja.html) created by Dr. Takamura from Institute of Science Tokyo.

<p align="center">
  <img src="/ksksk/images/negall.png" width="800">
</p>

# other basic functions to implement

1. classify by era and lyricist
    - most common words in each era and or by each lyricist
    - their favorite (?特徴的な) words. for example, i’m pretty sure entertainment~tree era fukase was the only one who used the word “sensou”
    - their 単語感情極性値 scores by era, by lyricist, and by era AND lyricist
2. correcting the *frequency* value by removing repeat words from a single song
    - for example
        - all 9 instances of「サイクル」come from 生物学, so the corrected value should be **1**
        - 「平和」both appears a lot in multiple songs AND appears frequently in single songs (sekai heiwa, love the warz), making it difficult to interpret its true frequency across their entire discography
3. removing redundant words (and english words for the time being)
    - probably gonna hand pick this one. it mostly occurs in the noun list anyway… the other lists look pretty clean
4. omake - you’ll have to hand pick this one as well
    1. comparisons of special words
        - seasons
        - nature
        - feelings
        - pronouns (boku vs bokura, minna/anata/kimi/omae)
        - organs??
    2. how many songs each member has written
       (seriously???)
    

# how the results are presented

idk… an interactive webpage sounds nice but that’s a lot of extra work.

right now, at least make wordclouds and or graphs.


use this list of words (words only) → search against masterdoc, increment by 1 for the first instance of each word in each row → export new word-frequency doc

# discussion

## 总览的数据处理

1. word pruning
    - 动词
      <br>
        <img src="/ksksk/images/doushi.png" width="200">
        - 主观的删除了大部分完全由假名构成的动词，因为这些词主要用于修饰汉字动词（例如「勝てる」「質問する」）
    - 形容词
      <br>
        <img src="/ksksk/images/adjs.png" width="500">  
        - 删除了「ない」「いい」这种词，虽然也很重要但不具有代表/特征性
    - 名词
        - 删除了无意义词，例如ん、よう、の、こと
        - 去掉了英文单词
            - 英文也可以单独拿出来分析
        - 去掉了katakana语
            - katakana语单独拿出来，按照时代分析！！
        - 删除了「君」「僕」等pronouns
            - pronouns单独拿出来，按照时代分析！！
