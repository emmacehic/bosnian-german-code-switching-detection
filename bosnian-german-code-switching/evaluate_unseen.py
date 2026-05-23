import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Loading the unseen samples
data = pd.read_csv("unseen_examples.csv")

# Same German word list as in the baseline
german_words = [
    "aber", "adresse", "adresu", "alles", "anerkennunga", "angekommen",
    "anmeldung", "anmeldunga", "antrag", "arbeit", "arbeiten", "arzt",
    "aufenthalt", "aufstehen", "aus", "ausland", "ausländerbehörde",
    "bericht", "bescheinigung", "beschäftigt", "bitte", "brot",
    "bürgeramt", "büro", "deutschland", "dokument", "dokumente",
    "ehrlich", "einkaufen", "essen", "fertig", "finde", "früh",
    "früher", "funkcioniše", "gehe", "gehen", "gesehen", "glaube",
    "heimat", "ich", "jobcentera", "komm", "komme", "kommen",
    "kompromis", "kämpfen", "lernen", "morgen", "müde", "nachricht",
    "papirologije", "perfekt", "rathaus", "rechnung",
    "regen", "sag", "schlafen", "schlüssel", "schon", "schreiben",
    "schwer", "sein", "sistem", "später", "steuer", "stresa",
    "stressig", "tag", "telefonieren", "telefonirati", "telefonnummer",
    "tempo", "termin", "terminima", "teuer", "träum", "verkehr",
    "viel", "vorbei", "wegen", "weiter", "überstunden"
]

# Same other word list as in the baseline
other_words = ["[name]", "[place]", "berlinu", "bukinga", "email",
"from", "id", "laptop", "mail", "meeting", "ok"]

# Simple prediction function which checks if a word is in one of the lists
def predict(word):
    word = str(word).lower()

    # If the word has numbers or is a number, labelling it as "O"
    if any(char.isdigit() for char in word):
        return "O"
    
    # Labelling it as "O" if its on the other_words list
    if word in other_words:
        return "O"
    
    # Labelling it as "DE" if its a German word
    if word in german_words:
        return "DE"
    
    # For everything else I assume that its Bosnian (BS)

    return "BS"

# Applying the prediction function to every word in the unseen dataset
# The predicted label is saved in a new column called "prediction"
data["prediction"] = data["word"].apply(predict)

# Printing the overall accuracy on the unseen examples
# Accuracy shows how many word labels were predicted correctly
print("Accuracy on unseen examples:")
print(accuracy_score(data["label"], data["prediction"]))

# Printing a more detailed evaluation with precision, recall, and F1-score
# zero_divison=0 avoids warnings if one class has no predicted examples.
print("\nDetailed results:")
print(classification_report(data["label"], data["prediction"], zero_division=0))

# Saving the unseen examples together with the model predictions
# This makes it possible to inspect which words were classified correctly or incorrectly.
data.to_csv("unseen_results.csv", index=False)

print("\nDone! Check unseen_results.csv")
