#!/usr/bin/env python
# coding: utf-8



import instaloader as instaloader
import os
def download_datafromprofile(username_targetprofile):
    loader = instaloader.Instaloader()
    loader.login("jiosache", "amlucky123")
    loader.download_profile(username_targetprofile, profile_pic=True, profile_pic_only=False, fast_update=False, download_stories=False, download_stories_only=False, download_tagged=False, download_tagged_only=False, post_filter=None, storyitem_filter=None)
    return 0
def data_separate(username_targetprofile):
    pathadress = username_targetprofile+('/')
   
    imgfiles = [] 
    for entry in os.listdir(pathadress):
        if os.path.isfile(os.path.join(pathadress, entry)): 
            if(entry.split(".")[-1]=="jpg"):
                imgfiles.append(entry)
    return imgfiles;

