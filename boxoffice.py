import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv("tmdb_5000_movies.csv", low_memory=False)
movies.head()
movies.info()

# this does same basic data cleaning to remove rows with missing or zero budget/revenue
movies = movies[(movies["budget"] > 0) & (movies["revenue"] > 0)]
movies = movies.dropna(subset=["runtime", "popularity", "vote_average"])

# this extracts the release year
movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
movies["budget_per_min"] = movies["budget"] / movies["runtime"]
movies["genre_count"] = movies["genres"].apply(
    lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
)
movies["decade"] = (movies["release_year"] // 10) * 10

# extract genre
all_genres = set()
for row in movies["genres"]:
    try:
        genres = [g["name"] for g in ast.literal_eval(row)]
        all_genres.update(genres)
    except:
        pass
# Initialize a column for each genre
for genre in all_genres:
    movies[f"genre_{genre}"] = 0

# Fill in 1 if the movie has that genre
for i, row in movies.iterrows():
    try:
        genres = [g["name"] for g in ast.literal_eval(row["genres"])]
        for g in genres:
            movies.loc[i, f"genre_{g}"] = 1
    except:
        pass

features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "budget_per_min",
    "genre_count",
] + [col for col in movies.columns if col.startswith("genre_")]

X = movies[features]
y = movies["revenue"]

# Apply log transform to reduce skew
movies["log_revenue"] = np.log1p(movies["revenue"])
y = movies["log_revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
# Convert log-revenue predictions back to real revenue for interpretability
preds_actual = np.expm1(preds)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), preds_actual))
print("RMSE:", rmse)
print("R^2 (on log scale):", r2_score(y_test, preds))

sns.scatterplot(x=y_test, y=preds)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Box Office Revenue")
plt.show()

corr = movies[["budget", "popularity", "runtime", "vote_average", "revenue"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
