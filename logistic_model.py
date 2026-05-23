import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

# Loading my annotated dataset (words + labels)
data = pd.read_csv("bosnian_german_code_switching_labeled.csv")

# Function to turn each word into features
# Tried to keep this as simple as possible
def make_features(word):
    word = str(word).lower()

    return {

        # The word itself so the model can learn common words directly
        # words like "ich" or "ja", appear often and the model can just memorize them
        "word": word,

        # Included these as simple signals, but not very strong on their own
        # For example Nouns in German are capitalized, and in Bosnian there is a less strict capitalization
        "first_letter": word[0],
        "last_letter": word[-1],

        # Suffixes as the probably most important feature
        # German verb endings (-en) vs Bosnian verb endings (-ti)
        # The endings help to distinguish them
        "last_2_letters": word[-2:],
        "last_3_letters": word[-3:],

        # Small additional signal
        # German has sometimes a bit longer words (compounds)
        "length": len(word),

        # Checking if it contains numbers
        # Helps to identify "Other (O)" Tokens
        "has_number": any(char.isdigit() for char in word),

        # Umlauts only appear in German
        # So if a word contains one its very likely German
        "has_umlaut": any(char in word for char in "äöüß")
    }

# Converting all words into feature dictionaries
X = [make_features(word) for word in data["word"]]

# Correct labels (BS / DE / O)
y = data["label"]

# Creating a simple logistic regression model
# Used a pipeline here, so the features get converted automatically
model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000))

# Using cross-validation, instead of a single train/test split
# Since my dataset is small, this should give more reliable results
predictions = cross_val_predict(model, X, y, cv=5)

# Printing evaluation results
print("Accuracy:")
print(accuracy_score(y, predictions))

print("\nDetailed results:")
print(classification_report(y, predictions))

# Saving my predictions to a csv to inspect them
data["logistic_prediction"] = predictions
data.to_csv("logistic_results.csv", index=False)

print("\nDone! Check logistic_results.csv")