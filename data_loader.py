import pandas as pd
import numpy as np

class BookVectorDataset:
    """
    Member 1 Module: Data Loading, Preprocessing, and Vectorization into R^20 Space.
    Reads the CSV dataset and extracts the 20-dimensional numerical feature matrix.
    """
    def __init__(self, csv_path="novels_vector_dataset_300.csv"):
        self.csv_path = csv_path
        self.df = None
        self.feature_columns = []
        self.feature_matrix = None
        self.load_and_vectorize()

    def load_and_vectorize(self):
        self.df = pd.read_csv(self.csv_path)
        self.feature_columns = [
            col for col in self.df.columns 
            if col.startswith("Genre_") or col.startswith("Attr_")
        ]
        # Convert the 20 feature columns into a 2D NumPy matrix of shape (300, 20)
        self.feature_matrix = self.df[self.feature_columns].to_numpy(dtype=np.float64)

    def find_book_index(self, query_title):
        """Finds book index by exact or partial case-insensitive title match."""
        query = query_title.strip().lower()
        titles_lower = self.df["Title"].str.lower()
        
        # Try exact match first
        exact_matches = self.df[titles_lower == query]
        if not exact_matches.empty:
            return int(exact_matches.index[0])
            
        # Try partial substring match
        partial_matches = self.df[titles_lower.str.contains(query, na=False, regex=False)]
        if not partial_matches.empty:
            return int(partial_matches.index[0])
            
        return None

    def get_book_vector(self, index):
        """Returns the 1D NumPy feature vector (in R^20) for a given book index."""
        return self.feature_matrix[index]

    def get_book_metadata(self, index):
        """Returns dictionary of book details."""
        row = self.df.iloc[index]
        return {
            "Book_ID": int(row["Book_ID"]),
            "Title": row["Title"],
            "Author": row["Author"],
            "Year": int(row["Publication_Year"]),
            "Category": row["Primary_Category"]
        }
