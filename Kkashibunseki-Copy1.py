#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


# In[2]:


list_df = pd.DataFrame(columns=['曲名','歌詞'])


# In[ ]:


base_url = 'https://www.uta-net.com'
url = 'https://www.uta-net.com/artist/9699/'
user_agent="""Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"""
header = {'User-Agent': user_agent}

response = requests.get(url,headers=header)

soup = BeautifulSoup(response.text, 'lxml')

links = soup.find_all('td', class_='sp-w-100')
for link in links:
    a = base_url + (link.a.get('href'))
    response = requests.get(a)
    soup = BeautifulSoup(response.text, 'lxml')
    song_name = soup.find('h2').text
        
    song_kashi = soup.find('div', id="kashi_area")
    song_kashi = song_kashi.text
        
    time.sleep(1)
        
    #index=(list_df.columns).T
    item = pd.DataFrame([[song_name, song_kashi]], columns=['name', 'kashi'])
    
    list_df = pd.concat([list_df, item], ignore_index=True)
    
    #list_df = list_df.append(item)


# In[ ]:





# In[ ]:


base_url = 'https://www.uta-net.com'
url = 'https://www.uta-net.com/artist/9699/'
user_agent="""Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"""

header = {'User-Agent': user_agent}

response = requests.get(url,headers=header)

soup = BeautifulSoup(response.text, 'lxml')

links = soup.find_all('td', class_='sp-w-100')
listy = pd.DataFrame(columns=['name', 'kashi'])

for link in links:
    a = base_url + (link.a.get('href'))
    response = requests.get(a)
    soup = BeautifulSoup(response.text, 'lxml')
    
    song_name = soup.find('h2').text
    song_kashi = soup.find('div', id="kashi_area").text
    song_lyricist = soup.find('a', itemprop='lyricist').text
    
    time.sleep(1)  
    
    item = pd.DataFrame([[song_name, song_kashi, song_lyricist]], columns=['name', 'kashi', 'lyricist'])
    
    listy = pd.concat([listy, item], ignore_index=True)


# In[ ]:


clean1 = listy.replace(["深瀬慧","藤崎彩織","中島真一"],["Fukase","Saori","Nakajin"])
clean1


# In[ ]:


clean2 = clean1.sort_values(by=['lyricist'])


# In[ ]:


def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]
clean3 = filter_rows_by_values(clean2, "lyricist", ["Saori","Nakajin"])


# In[ ]:


clean3.head()


# In[ ]:


pd.read_csv("skowclean.csv")


# In[ ]:


pip install janome


# In[ ]:


pip install wordcloud


# In[7]:


from janome.tokenizer import Tokenizer
import pandas as pd
import collections
df_file = pd.read_csv('skow.csv', encoding='utf-8')
song_lyrics = df_file['kashi'].tolist()
t = Tokenizer()
results = []
for s in song_lyrics: 
    result = [token.part_of_speech.split(',')[0] for token in t.tokenize(s)]
    results.extend(result)
c = collections.Counter(results)


# In[14]:


tangotype = c.most_common()


# In[15]:


typefreq = pd.DataFrame(tangotype, columns=['type','howmany'])


# In[3]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:


import matplotlib as mpl
print(mpl.matplotlib_fname())


# In[ ]:


reset kernel


# In[ ]:


typefreq


# ??

# In[ ]:


font_paths = fm.findSystemFonts()

# Get the font names safely by skipping faulty fonts
font_names = []
for font in font_paths:
    try:
        font_name = fm.FontProperties(fname=font).get_name()
        font_names.append(font_name)
    except RuntimeError as e:
        print(f"Error loading font {font}: {e}")

# Print all available font names
print(font_names)


# In[16]:


plt.figure(figsize=(12, 6))
sns.set(font='Hiragino Sans')
sns.barplot(data = typefreq, x='type', y='howmany')


# In[5]:


from wordcloud import WordCloud


# In[77]:


df_file = pd.read_csv('skow.csv', encoding='utf-8')
song_lyrics = df_file['kashi'].tolist()
t = Tokenizer()
results = []
for s in song_lyrics: 
    result = [token.base_form for token in t.tokenize(s) if token.part_of_speech.split(',')[0] in ['名詞']]
    results.extend(result)
cc = collections.Counter(results)


# only count the first instance each word appears in a song.


# In[78]:


cc


# In[149]:


df = pd.DataFrame.from_dict(cc, orient='index', columns=['頻度']).reset_index()
df.columns = ['名詞', '頻度']
df = df.sort_values(by='頻度', ascending=False)
df.to_csv("skowms_all.csv")


# In[80]:





# In[ ]:


#fpath = '/Users/kadomiii/Library/Fonts/ipaexg.ttf'
text = ' '.join(results) 
wordcloud = WordCloud(background_color='white', colormap="Set3",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=800, height=600, max_words=500).generate(text)

#wordcloud = WordCloud(background_color='white', font_path=fpath, width=800, height=600, max_words=500).generate(text)
#wordcloud.to_file('./wasf.png')


# In[28]:


plt.figure(figsize=(12, 6), dpi=300)
sns.set(font='Hiragino Sans')
sns.barplot(data = verbs_graph, x='動詞', y='頻度')
plt.xticks(rotation=45, fontsize=10)
plt.show()


# In[76]:


plt.figure(figsize=(12, 6), dpi=300)
sns.set(font='Hiragino Sans')
sns.barplot(data = adjs_graph, x='adjective', y='count', palette='Pastel2')
plt.xlabel('形容詞', fontsize=12)
plt.ylabel('頻度', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()


# In[127]:


plt.figure(figsize=(12, 6), dpi=300)
sns.set(font='Hiragino Sans')
ax = sns.barplot(data = mss_graph, x='名詞', y='頻度', palette='GnBu_r')
ax.bar_label(ax.containers[0])
plt.show()


# In[24]:


verbs = pd.read_csv("skowds.csv")
verbs = verbs.sort_values(by='頻度', ascending=False)
verbs_graph = verbs.head(20)


# In[63]:


adjs = pd.read_csv("skowadj.csv")
adjs = adjs.sort_values(by='count', ascending=False)
adjs_graph = adjs.head(20)


# In[126]:


mss = pd.read_csv("skowms.csv")
mss = mss.sort_values(by='頻度', ascending=False)
mss_graph = mss.head(20)


# In[151]:


msall = pd.read_csv("skowms_all.csv")
msall = msall.sort_values(by='頻度', ascending=False)


# In[ ]:





# In[ ]:


vdata = verbs.set_index('動詞').to_dict()['頻度']


# In[36]:


adjdata = adjs.set_index('adjective').to_dict()['count']


# In[130]:


msdata = mss.set_index('名詞').to_dict()['頻度']


# In[152]:


msdata_all = msall.set_index('名詞').to_dict()['頻度']


# In[324]:


negcloud = filtered_nec_neg.set_index('名詞').to_dict()['頻度']


# In[351]:


poscloud = filtered_nec_pos.set_index('名詞').to_dict()['頻度']


# In[305]:


filtered_nec_neg.head()


# In[ ]:


# VERBS
#text = verbs['動詞'].values 

#verbs_l = verbs.values.tolist()

#wordcloud = WordCloud().generate(str(text))
wordcloud = WordCloud(background_color='white', colormap="Set2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(vdata)


# In[60]:


# ADJECTIVES

wordcloud = WordCloud(background_color='white', colormap="Pastel2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(adjdata)


# In[153]:


# NOUNS

wordcloud = WordCloud(background_color='white', colormap="Set2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(msdata_all)


# In[ ]:


# NOUNS ALL

wordcloud = WordCloud(background_color='white', colormap="Set2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(msdata)


# In[329]:


# NEG NOUNS ALL

wordcloud = WordCloud(background_color='white', colormap="Set2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(negcloud)


# In[361]:


# POS NOUNS ALL

wordcloud = WordCloud(background_color='white', colormap="Set2",font_path = '/Users/kadomiii/Library/Fonts/ipaexg.ttf', width=1500, height=1200, max_words=500).generate_from_frequencies(poscloud)


# In[362]:


wordcloud.to_file('./posall.png')


# In[ ]:


verbs_l


# In[ ]:


cc


# In[ ]:


meishi = pd.DataFrame(results)
#ds_sorted = freqq.sort_values("頻度")
meishi.head()


# In[ ]:


freqq = pd.DataFrame(cc_items, columns=["名詞","頻度"])
ds_sorted = freqq.sort_values("頻度")


# In[ ]:





# In[ ]:


ds_sorted.to_csv('skowds.csv')


# In[ ]:





# In[ ]:


pip install --upgrade pip 


# In[ ]:


pip install --upgrade Pillow


# In[133]:


pd.set_option('display.unicode.east_asian_width', True)

pndic = pd.read_csv(r"http://www.lr.pi.titech.ac.jp/~takamura/pubs/pn_ja.dic",
                    encoding="shift-jis",
                    names=['word_type_score'])
print(pndic)


# In[135]:


pndic


# In[138]:


pndic[['word', 'hiragana', 'type', 'score']] = pndic['word_type_score'].str.split(':', expand=True)
pndic = pndic.drop(columns=['word_type_score'])

pndic['score'] = pndic['score'].astype(float)


# In[139]:


pndic


# In[364]:


nouns_pndic = pndic[pndic['type'] == '名詞'].copy()
nouns_pndic['score'] = nouns_pndic['score'].astype(float)
mergedic = pd.merge(msall, nouns_pndic[['word', 'score']], left_on='名詞', right_on='word', how='left')
mergedic = mergedic.drop(columns=['word']).rename(columns={'score': '感情スコア'})


# In[365]:


noun_emotion = mergedic.dropna()
len(noun_emotion)


# In[177]:


noun_emotionCleaned = noun_emotion.drop_duplicates(subset=['名詞'])


# In[182]:


len(noun_emotionCleaned)
noun_emotionCleaned.head()


# In[167]:


noun_emotionCleaned.to_csv('noun_emotionCleaned.csv')


# In[181]:


newEmotions = noun_emotionCleaned.drop(columns=['頻度'], inplace=True)
newEmotions


# In[297]:


plt.figure(figsize=(20, 12))
sns.scatterplot(data=noun_emotionCleaned, x='名詞', y='感情スコア', hue='感情スコア', palette='hls', s=100, alpha=0.6)

plt.gcf().set_facecolor('white')
plt.gca().set_facecolor('white') 
plt.title('感情分布')
plt.xlabel('名詞')
plt.ylabel('感情得分')
plt.xticks([])


# In[170]:


import plotly.express as px


# In[206]:


nec = pd.read_csv("noun_emotionCleaned.csv")


# In[208]:


nec.head()


# In[211]:


import numpy as np
np.bool = bool 


# In[264]:


sks = nec.head(1000)


# In[285]:


fig = px.scatter(sks, 
                 x='名詞', 
                 y='感情スコア', 
                 hover_name='名詞',  # This will show the word when hovered
                 title="感情vs名詞",
                 labels={"名詞": "名詞", "感情スコア": "感情スコア"})
fig.update_layout(
    #xaxis_range=[0, 100], 
    #yaxis_range=[-1, 1],
    width=1200,  
    height=800,
    plot_bgcolor='white',
    xaxis=dict(
        showticklabels=False,
        showgrid=True,  # Show gridlines on the x-axis
        gridcolor="lightgray",  # Gridline color
        gridwidth=1  # Gridline width
    ),
    yaxis=dict(
        showgrid=True,  # Show gridlines on the y-axis
        gridcolor="lightgray",  # Gridline color
        gridwidth=1  # Gridline width
    ),

)
fig.update_traces(marker=dict(size=10, color='turquoise', opacity=0.5))


# In[347]:


# now let's find the words with emotion scores > 0.98
filtered_nec_pos = nec[nec['感情スコア'] > 0.98]
filtered_nec_pos = filtered_nec_pos[filtered_nec_pos.名詞 != '上水']
filtered_nec_pos


# In[331]:


# let's look at an extreme. what about < -0.99?
filtered_nec_neg = nec[nec['感情スコア'] < -0.99]


# In[273]:


fig = px.scatter(
    sks,
    x="名詞",
    y="Index",
    size="感情スコア",  # Example size
    color="感情スコア",
    color_continuous_scale="RdBu"
)


# In[186]:


fig


# In[ ]:


pndic.to_csv('kanjoujisho.csv')


# In[ ]:


englishdic = pd.read_csv(r"http://www.lr.pi.titech.ac.jp/~takamura/pubs/pn_en.dic",
                    encoding="shift-jis",
                    names=['word_type_score'])
print(englishdic)


# In[ ]:


englishdic.to_csv('pn_en.csv')


# In[ ]:


conda activate wordcloud_env
conda install numpy matplotlib pillow wordcloud


# In[ ]:




