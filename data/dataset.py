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
        
    def split_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training, validation, and test sets.
        The training set contains 80% of the ratings, the validation set contains 10% of the ratings,
        and the test set contains 10% of the ratings.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): _description_
        """
        
        # Get 80% of the ratings for training
        train_size = int(0.8 * self.rating_count)
        val_size = int(0.1 * self.rating_count)
        
        # Shuffle the user rating dataframe
        shuffled_df = self._user_rating_df.sample(frac=1, random_state=8535).reset_index(drop=True)
        
        # Split the shuffled dataframe into training, validation, and test sets
        train_df = shuffled_df.iloc[:train_size]
        val_df = shuffled_df.iloc[train_size:train_size + val_size]
        test_df = shuffled_df.iloc[train_size + val_size:]
        
        assert len(train_df) + len(val_df) + len(test_df) == self.rating_count, "Dataset split error: Sizes do not match."
        
        train_pivot = train_df.pivot(index='user_id', columns='item_id', values='rating')
        all_items = range(1, self.movie_count + 1)
        train_pivot = train_pivot.reindex(columns=all_items, fill_value=0)

        return train_pivot, val_df, test_df
        