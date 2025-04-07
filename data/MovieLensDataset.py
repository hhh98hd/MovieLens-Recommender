import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MovieLensDataset:
    def __init__(self, dir : str):
        # user_id | item_id | rating | timestamp
        self._user_rating_df = pd.read_csv(dir + "u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

        # user id | age | gender | occupation | zip code
        self._user_info_df = pd.read_csv(dir + "u.user", sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zip_code"])
        # One-hot encode the occupation and gender columns
        self._user_info_df = pd.get_dummies(self._user_info_df, columns=['gender'])
        self._user_info_df = pd.get_dummies(self._user_info_df, columns=['occupation'], prefix='occ')
        # Normalize the age column to be between 0 and 1
        self._user_info_df['age'] = MinMaxScaler().fit_transform(self._user_info_df[['age']])
      
        # item id | movie title | release date | video release date || IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
        self._movie_info_df = pd.read_csv(dir + "u.item", 
                                sep="|", header=None,
                                names=["movie_id", "movie_title", "release_date", "url",
                                       "unknown", "Action", "Adventure", "Animation", "Children's", 
                                       "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
                                       "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
        
