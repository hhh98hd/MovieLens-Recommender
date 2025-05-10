import pandas as pd

from util import measure_execution_time, measure_memory

class MovieLensDataset:
    @measure_execution_time
    def __init__(self, dir : str):   
        # user_id | item_id | rating | timestamp
        self._user_rating_df = pd.read_csv(dir + "ratings.csv", sep=",", header=0)
        
        # movid_id | title | genres
        self._movie_df = pd.read_csv(dir + "movies.csv", sep=",", header=0)
        all_genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        # One-hot encoding of genres
        for genre in all_genres:
            self._movie_df[genre] = self._movie_df["genres"].apply(lambda x: 1 if genre in x else 0)
        self._movie_df.drop(columns=["genres"], inplace=True)
        
        def extract_year(title: str) -> int:
            """Extract the year from the title string."""
            if "(" in title and ")" in title:
                year = title.split("(")[-1].split(")")[0]
                if year.isdigit():
                    return int(year.strip())
                else:
                    return 0
            return 0
        self._movie_df["year"] = self._movie_df["title"].copy().apply(extract_year)
        avg_year = int(round(self._movie_df["year"].mean()))  # Ensure avg_year is int
        def year_to_timestamp(year: int) -> int:
            """Convert the year to a timestamp."""
            if year > 0:
                return pd.Timestamp(year=year, month=1, day=1).timestamp()
            else:
                return pd.Timestamp(year=avg_year, month=1, day=1).timestamp()
        self._movie_df["year"] = self._movie_df["year"].apply(year_to_timestamp)
        
        self.rating_count = self._user_rating_df.shape[0]
        self.user_count = self._user_rating_df["userId"].nunique()
        self.movie_count = self._movie_df.shape[0]
        
        self.user_ids = self._user_rating_df['userId'].astype('category').cat.codes
        self.movie_ids = self._user_rating_df['movieId'].astype('category').cat.codes
        
        self.userid_to_idx = { user_id: i for i, user_id in enumerate(self._user_rating_df['userId'].unique()) }
        self.movieid_to_idx = { movie_id: i for i, movie_id in enumerate(self._user_rating_df['movieId'].unique()) }
        
        print("MovieLens 20M dataset loaded successfully.")
        print(f"Rating count: {self.rating_count}")
        print(f"Movie count: {self.movie_count}")
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
    
        return train_df, val_df, test_df
        