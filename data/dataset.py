import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MovieLensDataset:
    def __init__(self, dir : str):
        # Load dataset metadata
        info = pd.read_csv(dir + "u.info", sep=' ', header=None)
        self.user_count = int(info.iloc[0, 0])
        self.movie_count = int(info.iloc[1, 0])
        self.rating_count = int(info.iloc[2, 0])
        
        # user_id | item_id | rating | timestamp
        self._user_rating_df = pd.read_csv(dir + "u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

        # user id | age | gender | occupation | zip code
        self._user_info_df = pd.read_csv(dir + "u.user", sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zip_code"])
        # One-hot encode the occupation and gender columns
        self._user_info_df = pd.get_dummies(self._user_info_df, columns=['gender'])
        self._user_info_df = pd.get_dummies(self._user_info_df, columns=['occupation'], prefix='occ')
        # Normalize the age column to be between 0 and 1
        self._user_info_df['age'] = MinMaxScaler().fit_transform(self._user_info_df[['age']])
      
        # item id | movie title | release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
        self._movie_info_df = pd.read_csv(dir + "u.item", 
                                sep="|", header=None,
                                names=["movie_id", "movie_title", "release_date", "url",
                                       "unknown", "Action", "Adventure", "Animation", "Children's", 
                                       "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
                                       "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
        # Convert the release_date column to timestamp
        self._movie_info_df['release_date'] = pd.to_datetime(self._movie_info_df['release_date'], errors='coerce')
        self._movie_info_df['release_date'] = self._movie_info_df['release_date'].astype('int64') // 10**9  # Convert to seconds since epoch
        
        print("MovieLens dataset loaded successfully.")
        print(f"User count: {self.user_count}")
        print(f"Movie count: {self.movie_count}")
        print(f"Rating count: {self.rating_count}")
        print("\n")
        
    def get_rating_pivot(self) -> pd.DataFrame:
        """Get the user rating pivot table.
        The pivot table has user_id as rows and item_id as columns, with ratings as values.

        Returns:
            pd.DataFrame: The user rating pivot table.
        """
        
        rating_pivot = self._user_rating_df.pivot(index='user_id', columns='item_id', values='rating')
        return rating_pivot
        