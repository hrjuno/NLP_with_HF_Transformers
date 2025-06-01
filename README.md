<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Harjuno Abdullah

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I hate doing homework")
```

Result : 

```
[{'label': 'NEGATIVE', 'score': 0.9994533658027649}]
```

Analysis on example 1 : 

The sentiment analysis model accurately identifies the negative emotion in the sentence. The phrase “I hate doing homework” clearly expresses frustration or dislike, and the model responds with a strong confidence score. This shows the model’s strength in detecting straightforward emotional language, which can be useful for analyzing feedback, social media posts, or product reviews.


### 2. Example 2 - Topic Classification

```
# TODO
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Online courses helped me understand data science much better.",
    candidate_labels=["education", "technology", "finance"],
)
```

Result : 

```
{'sequence': 'Online courses helped me understand data science much better.',
 'labels': ['technology', 'education', 'finance'],
 'scores': [0.6963863372802734, 0.24675911664962769, 0.05685454607009888]}
```

Analysis on example 2 : 

The model correctly matches the sentence with the label "technology" even though "education" might seem like a close fit. This highlights how the model evaluates the words "data science" and "online courses" in relation to broader topics. Its ability to classify topics without needing prior training on them makes it very flexible for real-world categorization tasks.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "If you do this exercise every night, you will feel",
    max_new_tokens=20,
    num_return_sequences=2,
)
```

Result : 

```
[{'generated_text': 'If you do this exercise every night, you will feel the need to get back to work, as you can tell. You will feel your heart beat and'},
 {'generated_text': 'If you do this exercise every night, you will feel like you are doing something wrong.\n\n\nI have a lot to say about this exercise and'}]
```

Analysis on example 3 : 

The text generation pipeline continues the sentence in two different but coherent directions. One talks about feeling energized, while the other reflects uncertainty. This shows that the model can generate creative continuations that sound natural and are grammatically correct. Such capabilities are great for idea generation, story writing, or completing prompts in chatbots or writing assistants.

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Machine learning <mask> are becoming increasingly popular.", top_k=4)
```

Result : 

```
[{'score': 0.5227693319320679,
  'token': 16964,
  'token_str': ' algorithms',
  'sequence': 'Machine learning algorithms are becoming increasingly popular.'},
 {'score': 0.09647637605667114,
  'token': 7373,
  'token_str': ' techniques',
  'sequence': 'Machine learning techniques are becoming increasingly popular.'},
 {'score': 0.05672439932823181,
  'token': 2975,
  'token_str': ' applications',
  'sequence': 'Machine learning applications are becoming increasingly popular.'},
 {'score': 0.03835589066147804,
  'token': 4233,
  'token_str': ' technologies',
  'sequence': 'Machine learning technologies are becoming increasingly popular.'}]
```

Analysis on example 3.5 : 

The model predicts “algorithms” as the most likely word to complete the sentence, which is contextually accurate. Other options like “techniques” and “applications” also make sense. This demonstrates the model’s strong understanding of word usage in technical contexts. It’s especially useful for language completion tasks or suggestions in writing tools.
### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Juno, I study at Universitas Indonesia, Depok")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9983315),
  'word': 'Juno',
  'start': 11,
  'end': 15},
 {'entity_group': 'ORG',
  'score': np.float32(0.99091995),
  'word': 'Universitas Indonesia',
  'start': 28,
  'end': 49},
 {'entity_group': 'LOC',
  'score': np.float32(0.88748723),
  'word': 'Depok',
  'start': 51,
  'end': 56}]
```

Analysis on example 4 : 

The NER model successfully identifies the correct types of entities—person (PER), organization (ORG), and location (LOC)—from the sentence. The high confidence scores show that the model is reliable in recognizing named entities, which is valuable in applications like resume parsing, document tagging, or building search engines that understand people, places, and institutions.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the capital city of France?"
context = "France is a country in Western Europe. Its capital city is Paris, which is known for its art, gastronomy, and culture."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.9910958409309387, 'start': 59, 'end': 64, 'answer': 'Paris'}
```

Analysis on example 5 : 

The model finds the correct answer "Paris" with very high confidence. This shows that it can match the question with the relevant part of the context effectively. The ability to pull exact answers from a text passage is essential for building virtual assistants, educational tools, or automated help desks

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer("""
Indonesia is one of the most biodiverse countries in the world, home to thousands of species of flora and fauna found nowhere else. From the rainforests of Kalimantan to the coral reefs of Raja Ampat, the natural richness of the archipelago is unmatched. However, this incredible biodiversity is under threat due to rapid deforestation, illegal wildlife trade, and climate change. Many species are endangered, and ecosystems are being disrupted at an alarming rate. To combat this, local communities, environmental organizations, and governments must collaborate in creating sustainable conservation strategies. Education and awareness are also crucial in ensuring future generations understand the value of protecting nature. By adopting eco-friendly practices and supporting conservation programs, individuals can make a significant difference. The fight to preserve Indonesia’s natural heritage is not easy, but it is essential for maintaining the balance of life on Earth.
""")
```

Result : 

```
[{'summary_text': ' Indonesia is one of the most biodiverse countries in the world, home to thousands of species of flora and fauna found nowhere else . Many species are endangered, and ecosystems are being disrupted at an alarming rate . The fight to preserve Indonesia’s natural heritage is not easy, but it is essential for maintaining the balance on Earth .'}]

```

Analysis on example 6 :

The summarizer condenses the long paragraph into a shorter version while retaining key messages about biodiversity, threats, and conservation. It simplifies the original content without losing the overall meaning. This feature is highly useful for quickly understanding news articles, research reports, or long blog posts.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Saya hobi mendaki gunung")
```

Result : 

```
[{'translation_text': "J'aime grimper dans les montagnes."}]

```

Analysis on example 7 :

The model translates the Indonesian sentence into French fluently and naturally. The result preserves the meaning of the sentence while following French grammar rules. This translation tool is helpful for communication in multilingual settings and for learners who want to practice or check their translations.

---

## Analysis on this project

This project gives a hands-on introduction to different NLP tasks using Hugging Face Transformers. Each example highlights a specific language task—from analyzing emotions to generating text and translating languages. It shows how just a few lines of code can unlock powerful language understanding. By experimenting with real examples, users can better understand the capabilities and practical uses of transformer models in everyday problems and tech solutions.
