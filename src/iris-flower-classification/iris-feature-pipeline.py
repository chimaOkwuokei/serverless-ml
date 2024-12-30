import random
import pandas as pd
import hopsworks
import uuid

# Set to true initially
BACKFILL = False

### Synthetic Data Functions
def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    df = pd.DataFrame({
        "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
        "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
        "petal_length": [random.uniform(petal_len_max, petal_len_min)],
        "petal_width": [random.uniform(petal_width_max, petal_width_min)]
    })
    df['variety'] = name
    return df

def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df = generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # Randomly pick one of these 3 and write it to the feature store
    pick_random = random.uniform(0, 3)
    if pick_random >= 2:
        iris_df = virginica_df
    elif pick_random >= 1:
        iris_df = versicolor_df
    else:
        iris_df = setosa_df

    return iris_df

# Backfill or create new synthetic input data
if BACKFILL == True:
    iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
else:
    iris_df = get_random_iris_flower()

iris_df["uuid"] = [str(uuid.uuid4()) for _ in range(len(iris_df))]


# Authenticate with Hopsworks using your API Key
project = hopsworks.login()
fs = project.get_feature_store()

# Create and write to a feature group - primary keys
iris_fg = fs.get_or_create_feature_group(
    name="iris",
    version=1,
    primary_key=["sepal_length", "sepal_width", "petal_length", "petal_width","uuid"],
    description="Iris flower dataset project"
)

iris_fg.insert(iris_df)