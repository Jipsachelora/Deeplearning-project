#!/usr/bin/env python
# coding: utf-8

# In[4]:

from data_scrape import download_datafromprofile,data_separate
from cnn_NatureorCities import training_cnn,cnn_test_image,make_confusion_matrix
import os
def main():
    print('Enter the profile name to search:')
    username = input()
    #Commented out datafromprofile download function this can be removed and new public profiles can be tested.
    #download_datafromprofile(username)
    imgfiles=data_separate(username)
    imagepath = os.getcwd()
    print(imagepath)
    Pooled_interested_destinationtype=[]
    print("training CNN")
    #Training cnn is commented out as there is a trained model already. You can remove the comment and do a fresh training on new data 
    training_cnn() 
    for imagefile in imgfiles:
        imagepathname = imagepath+'/'+ username +'/'+imagefile
        print("analysis completed")
        Pooled_interested_destinationtype.append(cnn_test_image(imagepathname))
    nature_count=Pooled_interested_destinationtype.count('Nature')
    cities_count=Pooled_interested_destinationtype.count('Cities')
    decision= "NATURE" if nature_count>cities_count else "CITIES"
    print('The person is interested more in '+decision+'. Suggest some places based on database of destination') 			
main()    
    

