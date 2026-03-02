import os
import kagglehub
import pandas as pd

file_path = "Womens Clothing E-Commerce Reviews.csv"
output_path = "../data/raw/reviews.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

dataset_path = kagglehub.dataset_download(
    "nicapotato/womens-ecommerce-clothing-reviews"
)

csv_path = os.path.join(dataset_path, file_path)

df = pd.read_csv(
    csv_path,
    encoding="latin1",
    engine="python",          
    on_bad_lines="skip"
)

df.to_csv(output_path, index=False)
print(f"Dataset downloaded and saved to {output_path}")