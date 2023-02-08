# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:05:06 2023

@author: prith
"""




import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta






# get data
df =pd.read_excel(r'D:\Python_Working\Text analyitcs\rawfile\keyword.xlsx')
#df.columns.tolist()
df['date'] = df['date'].dt.date


keywords=st.sidebar.text_input('Keywords', 'Enter Keywords')

# date range
start_date = st.sidebar.date_input("Start date", value=datetime.now() - timedelta(days=1365))
end_date = st.sidebar.date_input("End date", value=datetime.now())
st.write("Selected date range:", start_date, "to", end_date)


# get sentiment value
sentiment1 = df['reviewer_sentiment'].unique()
# use multiselect for display sentment
selected_sentiment1 = st.sidebar.multiselect('Select Sentiment',sentiment1,sentiment1)

# get verified value
verified = df['verified'].unique()
# use multiselect for display verified
selected_verified = st.sidebar.multiselect('Select verified purchase or others',verified,verified)


# get product value
product = df['Product'].unique()
# use multiselect for display product
selected_product = st.sidebar.multiselect('Select Product',product,product)

 

dmax = df['reviewer_rating_'].max()
dmin = df['reviewer_rating_'].min()
rating = st.sidebar.slider('Select Rating',dmin,dmax,dmax)


# filter dataframe bases on sentment selected
if keywords =="Enter Keywords":
    df2 =df.copy()
else:
    df2 = df[df['reviewer_text'].str.contains(keywords, case=False, na=False)]
# filter dataframe bases on sentment selected
df2 = df2[df2['reviewer_sentiment'].isin(selected_sentiment1)]
# filter dataframe based on selected verified
df2 = df2[df2['verified'].isin(selected_verified)]
# filter dataframe based on selected product
df2 = df2[df2['Product'].isin(selected_product)]
# filter dataframe based on rating
df2 =df2[df2['reviewer_rating_']<=rating]
# filter dataframe based on date
df_filter=df2[(df2['date'] >= start_date) & (df2['date'] <= end_date)]

# group by dataframe with selected column only
df_filter1 = df_filter[['date', 'Count']].groupby(['date']).sum('Count')
# make date index to column
df3 = df_filter1.reset_index()
fig = px.bar(df3,x="date", y="Count", title="Daily review trend")
st.plotly_chart(fig, use_container_width=True)
if keywords =="Enter Keywords":
    st.write('Total review count is',len(df_filter))
else:
    st.write('Total review count' ,len(df_filter), 'as per keyword',keywords)

#plot.write_html(r'D:\Python_Working\Streamlit\rawfile\nps_gauageytd.html')


import plotly.graph_objects as go
import numpy





df_filter1 =df_filter[df_filter['reviewer_sentiment']=='Positive']
df_filter1 = df_filter1[['date', 'Count']].groupby(['date']).sum('Count')
df3 = df_filter1.reset_index()
df_date =df3['date'].tolist()
df_count =df3['Count'].tolist()

df_filter2 =df_filter[df_filter['reviewer_sentiment']=='critical']

df_filter2 = df_filter2[['date', 'Count']].groupby(['date']).sum('Count')
dfC3 = df_filter2.reset_index()
dfC_date =dfC3['date'].tolist()
dfC_count =dfC3['Count'].tolist()
  
plot = go.Figure()
 
plot.add_trace(go.Scatter(
    name = 'Positive Review',
    x = df_date,
    y = df_count,
    stackgroup='one'
   ))

plot.add_trace(go.Scatter(
    name = 'Critical Review',
    x = dfC_date,
    y = dfC_count,
    stackgroup='one'
   ))

plot.update_layout(title="Positive vs Critical Reviews")

st.plotly_chart(plot, use_container_width=True)


if sum(dfC_count)<sum(df_count):
    result ='Positive'
else:
    result ='Critical'
    
st.write('With ',sum(df_count),' positive reviews and ',sum(dfC_count),' critical reviews, the majority of feedback received is ',result)
  


import plotly.express as px

#df_filter.columns.tolist()
df_filter3 =df_filter[['Product', 'SentimentScore','reviewer_sentiment']]

df_filter3 = df_filter3.groupby(['Product','reviewer_sentiment']).sum('SentimentScore')
df_filter3 = df_filter3.reset_index()
df_filter3 = df_filter3.pivot_table(index='Product', columns='reviewer_sentiment', values='SentimentScore')
df_filter3.reset_index(inplace=True)

fig1 = px.bar(df_filter3, x=["Positive","critical"], y="Product", orientation='h',title='Product breakup in Sentiment')
#fig1.write_html(r'D:\Python_Working\Streamlit\rawfile\nps_gauageytd.html')
st.plotly_chart(fig1, use_container_width=True)


import plotly.express as px

#df_filter.columns.tolist()
df_filter4 =df_filter[['Product', 'Count','reviewer_rating']]

df_filter4 = df_filter4.groupby(['Product','reviewer_rating']).sum('Count')
df_filter4 = df_filter4.reset_index()
df_filter4 = df_filter4.pivot_table(index='Product', columns='reviewer_rating', values='Count')
df_filter4 = df_filter4.reset_index()



fig2 = px.bar(df_filter4, x=["1.0 out of 5 stars","2.0 out of 5 stars","3.0 out of 5 stars","4.0 out of 5 stars","5.0 out of 5 stars"], y="Product", orientation='h',title='Product breakup in Rating'
              )
#fig2.write_html(r'D:\Python_Working\Streamlit\rawfile\nps_gauageytd.html')
st.plotly_chart(fig2, use_container_width=True)



from PIL import Image
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt 
from collections import Counter
import pandas as pd
import re

nltk.download('stopwords')
nltk.download('punkt')


porter = PorterStemmer()
stop_words = set(stopwords.words('english')) 
NEW_Stopwords=['rt','RT','http','HTTP','I','We',
               'My','https','The','So','MY','u','and','also',
               'BoatNirvana','please','kindly','Such','name','lionsgate','Lionsgate']
stop_words.update(NEW_Stopwords)


df_filter['reviewer_text'] = df_filter['reviewer_text'].astype(str)
df_sent = ' '.join(df_filter['reviewer_text'])

df_word = word_tokenize(df_sent)


#len(word_tokens)


# only alphabhets
df_word = [word for word in df_word if word.isalpha()]

#Remove stop wordd
filtered_sentence = [w for w in df_word if not w in stop_words] 



# lower case
dff = [w.lower() for w in filtered_sentence]

# stem
#dff = [porter.stem(word) for word in dff]

comment_words = ' '

comment_words=comment_words.join(dff)

# comment_words = ' '
# for words in dff: 
#     comment_words = comment_words + words + ' '
    
#comment_words=re.sub(r'(?:^| )\w(?:$| )', ' ', comment_words).strip()

shortword = re.compile(r'\W*\b\w{1,2}\b')

comment_words=shortword.sub('', comment_words)

   

wordcloud = WordCloud(width = 1500, height = 1500, colormap="Oranges",
            background_color ='black',max_words=200,
                    min_font_size = 0).generate(comment_words)



fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)
st.write("Word Cloud of example text")












