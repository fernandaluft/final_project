# -*- coding: utf-8 -*-
"""scraping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ozb1SaVKYEDkefHT36cKixmmELxh8nsZ

## Scraping Module
"""

import os

class Scraping:
    def __init__(self, IN_COLAB, books_reviews_ds = False, sentiment_ds = False):
        if IN_COLAB is True: 
          self.base_path = "/content/"
        else: 
          self.base_path = "/workspaces/"
          
        self.books_reviews_ds = books_reviews_ds
        self.sentiment_ds = sentiment_ds

        # Create the directory if it doesn't exist
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)

        # Copy kaggle.json to the directory
        kaggle_json_src = self.base_path + 'final_project/config/kaggle.json'
        kaggle_json_dst = os.path.join(kaggle_dir, 'kaggle.json')
        os.system(f'cp {kaggle_json_src} {kaggle_json_dst}')

        # Set permissions for kaggle.json
        os.system(f'chmod 600 {kaggle_json_dst}')

        # Import the kaggle module
        import kaggle

    def kaggle_scrape(self):
        # Download the dataset using the kaggle CLI
        if self.books_reviews_ds is True: 
            os.system('kaggle datasets download -d mohamedbakhet/amazon-books-reviews')

        if self.sentiment_ds is True: 
            os.system('kaggle datasets download -d crisbam/imdb-dataset-of-65k-movie-reviews-and-translation')
            #kaggle.api.dataset_download_files()
        
        # Unzip the downloaded file
        os.system(f'unzip -q {self.base_path}/amazon-books-reviews.zip -d {self.base_path}final_project/data')
        os.system(f'rm {self.base_path}/amazon-books-reviews.zip')
        