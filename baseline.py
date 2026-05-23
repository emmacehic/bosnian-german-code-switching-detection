import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


# Loading the dataset (my annotated csv file words + labels)
data = pd.read_csv("bosnian_german_code_switching_labeled.csv")


# List of German words, which I manually collected from the dataset
# Using a simple rule-based approach instead of training a model
german_words = ["aber", "adresse", "adresu", "alles", "anerkennunga", "angekommen",
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


# Words that should be labeled as "O" (O means other, not bosnian or German)
# This includes English words, placeholders etc
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

# Applying the prediction function to every word in the dataset
data["prediction"] = data["word"].apply(predict)

# Evaluating the results
# Accuracy = how many prediction are exactly correct
print("Accuracy:")
print(accuracy_score(data["label"], data["prediction"]))

# More detailed evaluation (precision, recall and f1-score per class)
print("\nDetailed results:")
print(classification_report(data["label"], data["prediction"]))

# Saving the results into a new file so I can inspect them later
data.to_csv("baseline_results.csv", index=False)

print("Done! Check baseline_results.csv")