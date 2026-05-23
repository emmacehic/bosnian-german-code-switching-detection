
# Bosnian-German Code-Switching Detection

A small NLP project for detecting word-level language switching between
Bosnian and German using both rule-based and machine learning approaches.

## Quick Start

# Clone the repository

git clone https://github.com/emmacehic/bosnian-german-code-switching-detection.git

cd bosnian-german-code-switching-detection

# Install dependencies
pip install -r requirements.txt

# Run baseline model
python baseline.py

# Run logistic regression model
python logistic_model.py

# Run small unseen evaluation
python evaluate_unseen.py

Results will be saved as:
- baseline_results.csv
- logistic_results.csv
- unseen_results.csv

Note (Mac/Linux users):
- You may need to use python3 and pip3 instead of python and pip.



## Project Motivation

Code switching, the practice of mixing multiple languages within a single sentence, is very common among multilingual speakers. For example, Bosnian speakers living in German-speaking environments often naturally mix Bosnian and German in everyday communication.

For Bosnian-German diaspora text, code-switching creates a concrete NLP bottleneck, because many basic tools assume that a sentence is written in only one language. This makes preprocessing, search, corpus annotation, and later tasks such as translation or sociolinguistic analysis harder, because German words, Bosnian words, English tokens, names, and adapted mixed forms appear together in the same informal text.

This project addresses this bottleneck by building a small word-level language identification prototype. The system labels each token as Bosnian (BS), German (DE), or Other (O), which can be used as a first preprocessing or annotation step for mixed Bosnian-German text.

The goal of this project is to build a system that can identify the language of each individual word in such mixed sentences.

Instead of directly using complex models, this project explores a more fundamental question:

- How far can a simple rule-based system go?
- What does a basic machine learning model learn beyond that?
- Which linguistic features are actually useful for this task?



## Project Scope

This project covers multiple core aspects of NLP.

- Data: Creation of a manually annotated dataset of Bosnian-German code-switching.

- Learning: Comparison between a rule-based system and a machine learning model.

- Evaluation: Quantitative analysis using accuracy and class-level metrics.

The goal is to build a small working prototype for word-level language identification in Bosnian-German code-switched text, while also analyzing the strengths and weaknesses of different approaches.



## Dataset

A custom dataset was created and annotated at the word level.

Each word is labeled as:
- BS: Bosnian
- DE: German
- O: Other (e.g. English words, numbers etc)

### Data collection

The data was collected from a combination of:

- Private WhatsApp conversations (family group chats), which reflect natural and informal language use

- Public online discussions, from the platforms Facebook and Reddit, where members of the Bosnian diaspora in Germany, Switzerland, and Austria discuss topics like daily life, work, and immigration

All data was anonymized (e.g. replacing names with [NAME]) to preserve privacy.

### Characteristics

The dataset aims to reflect realistic, informal communication, including:

- Mixed grammar and sentence structures
- Frequent code-switching within sentences
- Borrowed or adapted words between languages
- Everyday vocabulary related to life abroad

### Example

Ich komme kasnije kući, [NAME] me zadrzao na poslu.

Jel mozeš kupiti Brot kad ideš iz Arbeit?

Moram morgen früh aufstehen, imamo Termin kod doktora.



## Annotation Guidelines
To create the dataset, each word was manually labeled as BS, DE, or O. Since code-switching data can be ambiguous, I created a few simple guidelines to keep the annotation consistent.

### Label Definitions
- BS (Bosnian): Words that are clearly a part of bosnian vocabulary
- DE (German): Words that are clearly German, including common nouns, verbs and expressions
- O (Other): Words that do not belong to either language, such as:
    - English words (e.g. email, meeting, etc.)
    - Names and placeholders (e.g. [NAME])
    - Numbers or tokens containing digits

### Annotation Decisions
Some cases required more careful decisions:
- Mixed or adapted words: Words like 'Anmeldunga' were labeled as DE, since they are based on a German root, even if they include Bosnian endings.
- Ambiguous words: Words that exist in both languages (e.g. problem, ja) were labeled based on their most likely usage, since no sentence-level context was used in the model.
- Lowercase / informal spellings: Informal writing was kept as it is, and labels were assigned based on meaning rather than spelling.

### Limitations of Annotation
Since the dataset was annotated manually and without full context modeling, some labels may still be ambiguous or subjective.

However, the goal was to create a consistent and realistic dataset, rather than a perfectly clean one, in order to better reflect on real-world code-switching.



## Approach 1: Rule-Based Baseline

The first approach is a simple rule-based system.

It uses:
- A manually collected list of German words
- A small list of tokens labeled as "Other"
- A few simple rules:
    - Words containing numbers: "O"
    - Known German words: "DE"
    - Everything else: "BS"

This approach is easy to understand and works quite well for words that are already included in the lists. 

One important limitation of this baseline is that the German word list was collected manually from the same small dataset that is later evaluated. This means that the very high baseline score should not be interpreted as true generalization to new data. The baseline is useful as a transparent comparison point, but it partly memorizes known vocabulary rather than learning broader patterns of Bosnian-German code-switching.

One reason it performs relatively well is that the dataset is quite small (around 100 sentences) and contains many repeated and clearly identifiable words. In this type of code-switching, common German words appear frequently, so a manually collected list can already capture almost everything of the data. Because of this it is able to achieve good performance, even though it does not actually learn any deeper patterns. With a larger dataset, the advantage of the learning-based approach would likely become more clear.

However, it has some clear limitations. It depends heavily on how complete the word lists are, so it struggles with:
- New or unseen German words
- Informal spelling variation
- Mixed or adapted words.

Overall, this baseline is useful because it shows how far simple rules can go, and it gives a clear point of comparison for the machine learning approach.



## Approach 2: Logistic Regression Model

The second approach uses a machine learning model with handcrafted features.

Instead of relying on a predefined word list, each word is converted into a set of features. 
These features include:
- The word itself
- First and last letters
- Word endings (last 2-3 characters)
- Word length
- Presence of numbers
- Presence of German-specific characters (e.g. ä, ö, ü)

These features are then used to train a Logistic Regression model. I used a DictVectorizer to transform the features and applied then a 5-fold cross validation to get more reliable results, since the dataset is small.

Compared to the rule-based approach, this model does not just memorize words, but tries to learn patterns, for example that German words often end in -en or that umlauts are a strong signal for German. This makes it more flexible, especially when dealing with words that were not explicitly seen before.



## Results and Observations

Both approaches are evaluated using:
- Accuracy
- Precision, Recall, and F1-score per class

The rule-based model achieves a very high accuracy (~0.998), with almost perfect scores across all classes. However, this result should be interpreted carefully, because the rule-based word list was created from the same dataset. Therefore, the baseline mainly shows how far a transparent vocabulary-based approach can go on this specific dataset, rather than how well it would generalize to completely unseen Bosnian-German text.

At first, this looks extremely strong, but the reason for this high performance is that the dataset is relatively small and contains many repeated words. Since the model relies on a manually created list of German words, it is able to correctly classify most of the data simply by matching known vocabulary.

The logistic regression model achieves a lower overall accuracy (~0.95), but behaves differently:
- It performs very well on Bosnian (BS) words (F1 ≈ 0.97)
- It is less accurate on German (DE) words (F1 ≈ 0.86), especially on recall
- It struggles the most with the "O" (Other) category (recall ≈ 0.62)

In addition to the original evaluation, I also tested the rule-based baseline on a small unseen evaluation set. On this unseen set, the baseline achieved an accuracy of approximately 0.815. This lower result confirms that the very high score on the original dataset is partly due to the manually collected word list and does not fully reflect generalization to new examples.

Some key observations:
- The rule-based model performs extremely well on known vocabulary, but this is largely due to memorization rather than generalization.
- The logistic regression model generalizes better, especially by using suffix patterns and character features.
- Features like word endings and umlauts are strong signals for identifying German words.
- The task becomes much harder with informal language and mixed forms, which are very common in real communication.
- The unseen evaluation shows that unseen German words such as 'Krankenkasse', 'Gemeinde', 'Vertrag', and 'Spitalu' are difficult for the rule-based model, because they are not included in the manually created German word list.

Overall, the machine learning model is more flexible, but the rule-based model achieves higher accuracy on this specific dataset due to its structure.



## Small Unseen Evaluation

Because the rule-based baseline uses a manually collected German word list, I also added a small unseen evaluation set. The examples in this small set were generated with the help of an LLM and then manually checked and labeled by me. They were not used when creating the original dataset or the German word list.

The purpose of this evaluation is not to provide a statistically strong benchmark or to replace evaluation on naturally occurring data. Instead, it is used as a small stress test to check how the system behaves on new Bosnian-German code-switched examples, especially examples containing unseen German words, adapted forms, and tokens from other languages.

The LLM-assisted unseen set contains 27 manually checked word-level examples:
- 18 Bosnian tokens
- 7 German tokens
- 2 Other tokens

On this small LLM-assisted unseen set, the rule-based baseline achieves an accuracy of approximately 0.815. This is clearly lower than its score on the original dataset, which shows that the original result should not be interpreted as full generalization.

The unseen evaluation includes examples such as:
- new German administrative words, such as 'Krankenkasse', 'Gemeinde', and 'Vertrag'
- code-switching at different positions in the sentence
- English or other tokens, such as 'online' and 'meeting'
- adapted or regionally influenced forms, such as 'Spitalu'

The errors show the main weakness of the rule-based approach. It correctly identifies known German words such as 'Termin', 'Ich', and 'Anmeldung', but it misclassifies unseen German words such as 'Krankenkasse', 'Gemeinde', 'Vertrag', and 'Spitalu' as Bosnian. It also misclassifies 'online' as Bosnian instead of Other.

This supports the main conclusion of the project: the rule-based baseline is transparent and useful, but its very high score on the original dataset partly reflects memorization of known vocabulary. For real-world use, the system would need a larger and more diverse dataset, better handling of unseen German words, and possibly character-level or context-aware features.



## Error Analysis

When looking more closely at the results, I noticed that both models make mistakes, but for different reasons.

The rule-based model performs mainly because it relies on a manually created list of German words. This works well for words that are included in the list, but it also shows its main weakness: it mostly just "memorizes" words. If a word appears in a slightly different form, for example with a different ending, the model often misclassifies it.

The logistic regression model behaves differently. It tries to learn patterns from features instead of memorizing words. Because of this, it can sometimes classify unseen words, but it also makes more mistakes in cases where patterns are unclear or ambiguous.

One important source of errors are words that exist in both languages.
For example:
- 'ja' can mean "yes" in German, but "I" in Bosnian
- 'problem' exists in both languages

Since the model looks at words individually and does not use context, it is very difficult to classify these correctly.

Another challenge are mixed or adapted words, which are common in real communication.
For example:
- Forms like 'Anmeldunga', where a German word is combined with a Bosnian ending.
These cases are very difficult because they do not fully belong to one language, and both models struggle with them.

In this dataset, both models actually classify 'Anmeldunga' correctly as German. The rule-based model succeeds because this specific form is included in the word list, while the logistic regression model is able to recognize patterns from the German root 'Anmeldung'. However this still highlights a limitation, because both approaches might struggle with similar mixed forms that were not seen before. This shows that handling real-world code-switching requires more than just memorizing words or simple patterns.

The model also struggles with the "O" (Other) category, which includes numbers, names or English words. This can be seen in the lower recall 
(around 0.62) This is likely, because this class is less frequent in the dataset, so the model does not learn it as well and sometimes confuses it with Bosnian or German.

In general, the logistic regression model performs best on Bosnian words, but has more difficulty distinguishing German words and especially the "O" category. This is likely because Bosnian is more frequent in the dataset.



## Limitations and Failure Cases

Even though both models achieve reasonable results, there are still several important limitations:
- Ambiguity: Some words belong to both languages or are very similar, which makes the classification difficult.
- No context: Each word is classified independently, without considering the sentence it appears in.
- Out-of-vocabulary words: The rule-based model cannot handle words that are not in its list.
- Small dataset: The models may not generalize well to other types of data.

More specifically, the logistic regression model still struggles with:
- Informal spelling variations
- Hybrid or mixed-language words
- Domain specific vocabulary

These limitations show that code-switching detection is not just about recognizing words, but also about handling ambiguity and variation in real language use.



## Ethical and Community Considerations
- This project works with informal diaspora language, which can include private or sensitive communication. For this reason, all personal names and locations were anonymized, and private source conversations are not redistributed without consent.

- The system should not be used to judge speakers' language quality or correctness. Code-switching is treated here as a normal and meaningful multilingual practice, not as an error.

-  Because the dataset is small and manually annotated, the model should be understood as a prototype rather than a general-purpose tool. Its main value is to support further corpus annotation, linguistic analysis, and future development of NLP tools for Bosnian-German mixed language.



## Reproducibility
This project is fully reproducible:
- The dataset is included in the repository
- All preprocessing and modeling steps are contained in the scripts
- Running these scripts will regenerate the results from scratch.

To reproduce the results:
python baseline.py
python logistic_model.py
python evaluate_unseen.py

The generated outputs will match the provided result files: 'baseline_results.csv', 'logistic_results.csv', and 'unseen_results.csv'.



## Project Structure
-- baseline.py
-- logistic_model.py
-- evaluate_unseen.py
-- bosnian_german_code_switching_labeled.csv
-- bosnian_german_code_switching_raw.txt
-- unseen_examples.csv
-- baseline_results.csv
-- logistic_results.csv
-- unseen_results.csv
-- README.md
-- requirements.txt



## Why compare these Approaches?
In this project, I wanted to compare two very different ways of solving the same problem.

The rule-based system relies on explicit knowledge, meaning that it uses manually defined word lists and simple rules. In contrast, the machine learning model tries to learn patterns directly from the data, without being told exactly what to look for.

By comparing these two approaches, I wanted to better understand:
- How far a simple rule-based system can go.
- Whether a machine learning model can capture patterns that rules miss.

This comparison is important because the goal of the project is not just to achieve the highest possible accuracy, but to understand how different methods behave on this task and where their strengths and weaknesses are.



## Possible Improvements 
If this project were extended further, the following steps could improve performance:
- Collect larger and more diverse dataset.
- Use character-level models, so unknown words are getting better handled.
- Experiment with transformer-based models, such as multilingual BERT.
- Use the surrounding words (context) instead of classifying each word on its own.



## Final Thoughts
This project is a small but very practical exploration of multilingual NLP in real-world settings. It highlights how even simple approaches can provide useful insights, while also showing the limitations that arise when dealing with informal, mixed-language data. The additional unseen evaluation also shows that future work should focus especially on generalization to new German words, adapted forms, and more diverse real-world Bosnian-German data.
