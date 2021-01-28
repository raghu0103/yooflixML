#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 01:20:34 2021

@author: Raghwendra Sharan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

videos_init_df = pd.read_csv("video.csv")
lkcategory_df = pd.read_csv("lkcategory.csv")
lksubcategory_df = pd.read_csv("lksubcategory.csv")
category_df = pd.read_csv("category.csv")
subcategory_df = pd.read_csv("subcategory.csv")
video_hash_df = pd.read_csv("videohash.csv")
hashtag_df = pd.read_csv("hashtag.csv")
channel_df = pd.read_csv("channel.csv")


category_df['category_name'] = category_df.join(lkcategory_df.set_index('primary_key'), on='lkvidecategory',how='left', lsuffix='_left', rsuffix='_right')['name']
subcategory_df['subcategory_name'] = subcategory_df.join(lksubcategory_df.set_index('primary_key'), on='lkvideosubcategory',how='left', lsuffix='_left', rsuffix='_right')['name']
video_with_channel_hashtag_df = videos_init_df.join(channel_df.set_index('primary_key'), on='eochannel',how='left', lsuffix='_left', rsuffix='_right')
video_with_channel_hashtag_category_df = video_with_channel_hashtag_df.join(category_df.set_index('eovideo'), on='primary_key',how='left', lsuffix='_left', rsuffix='_right')
video_with_channel_hashtag_category_subcategory_df = video_with_channel_hashtag_category_df.join(subcategory_df.set_index('eovideo'), on='primary_key',how='left', lsuffix='_left', rsuffix='_right')


features = ["title","channel_name","category_name","subcategory_name","description_left"]
for feature in features:
    video_with_channel_hashtag_category_subcategory_df[feature] = video_with_channel_hashtag_category_subcategory_df[feature].fillna('')
    

def combine_feature(row):
    try:
        return row["title"] + " " + row["description_left"] + " " + row["channel_name"] + " " + row["category_name"] + " " + row["subcategory_name"]
    except:
        print(row)

video_with_channel_hashtag_category_subcategory_df["combine_feature"] = video_with_channel_hashtag_category_subcategory_df.apply(combine_feature,axis=1)

df = video_with_channel_hashtag_category_subcategory_df
df.drop_duplicates(subset=['combine_feature'])

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combine_feature"])

cosine_model = cosine_similarity(count_matrix)
cosine_model_df = pd.DataFrame(cosine_model,index=df.title,columns=df.title)
#cosine_model_df.head()

#Saving model to pickle
pickle.dump(cosine_model_df, open('model.pkl','wb'))

def make_recommendations(video_user_likes):
    return cosine_model_df[video_user_likes].sort_values(ascending=False)[:10]


print(make_recommendations('Best Scenes of 3 Idiots '))