import datetime

import pandas as pd

class MovieLensDataset:

    def __init__(self, dir : str):
        # user_id | item_id | rating | timestamp
        self._user_rating = pd.read_csv(dir + "u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

        # user id | age | gender | occupation | zip code
        self._ser_info = pd.read_csv(dir + "u.user", sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zip_code"])
        
        # item id | movie title | release date | video release date || IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
        self._movie_info = pd.read_csv(dir + "u.item", 
                                sep="|", header=None,
                                names=["movie_id", "movie_title", "release_date", "url",
                                       "unknown", "Action", "Adventure", "Animation", "Children's", 
                                       "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
                                       "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
        
        