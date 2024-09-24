# Sentiment_Text_Classifier_based_on_distilBERT
 This Streamlit app generates a title, summary, and detects emotions with emojis using Hugging Face models.

This project is a Streamlit-based web application that analyzes user input text. It performs three tasks: generates a catchy title, provides a summary, and performs emotion classification with emojis. The app uses pre-trained NLP models from Hugging Face's Transformers for text summarization and emotion classification, along with NLTK for tokenization.

### Text Analyzer Web Application

This project is a **Streamlit-based web application** that analyzes user input text. It performs three tasks:
1. **Generates a catchy title** from the text.
2. **Provides a summary** of the text.
3. **Performs emotion classification** with emojis.

The app uses pre-trained **NLP models from Hugging Face's Transformers** for text summarization and emotion classification, and utilizes **NLTK** for tokenization.

---

### Project Overview:

This repository hosts a **text analysis web application** built with **Streamlit**. It allows users to input text and performs three main functions:

- **Title Generation**: Generates a concise, catchy title from the input text.
- **Text Summarization**: Provides a summarized version of the input text.
- **Emotion Classification**: Detects the top three emotions from the text and associates them with relevant emojis.

---

### Key Features:

1. **Streamlit UI**: The app provides a simple, interactive interface using **Streamlit**, where users input text and get instant feedback on analysis.
   
2. **Text Summarization**: The app uses a pre-trained **BART model** (`facebook/bart-large-cnn`) to generate concise summaries by extracting key points from longer text.

3. **Emotion Classification**: Emotion detection is powered by the **DistilRoBERTa model** (`j-hartmann/emotion-english-distilroberta-base`). It returns the top three emotions associated with the text, displayed alongside relevant emojis.

4. **Title Generation**: The **T5 model** (`t5-small`) generates a catchy title, limited to 10 words, with a period at the end.

5. **NLTK Integration**: The app uses **NLTK** for tokenizing text into sentences using `sent_tokenize`.

---

### Code Explanation:

#### Model Loading:
Three pre-trained models from Hugging Face are loaded for summarization, emotion classification, and title generation. These are cached using **Streamlit's `@st.cache_resource`** to avoid reloading them with every user interaction.

```python
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="  ")
    emotion_classifier = pipeline("text-classification", model="  ", top_k=None)
    title_generator = pipeline("summarization", model=" ", tokenizer="t5-small")
    return summarizer, emotion_classifier, title_generator
```

#### Title Generation:
The **T5 model** generates a short, catchy title. The title is cleaned to fit within 10 words, converted to lowercase, and ensures a period at the end.


#### Text Summarization:
The **BART model** reduces long text into concise summaries, limited between 30 and 150 tokens.



#### Emotion Classification:
Emotion detection is done with the **DistilRoBERTa model**. The top three emotions are displayed alongside relevant emojis.



#### UI and Interactivity:
The **Streamlit** interface is designed for simplicity. Users input text into a text area, and upon clicking "Analyze Text", the app generates a title, summary, and emotion analysis.

---

### Dependencies:

- **Streamlit**: For building the web interface.
- **Transformers**: For pre-trained NLP models from Hugging Face.
- **NLTK**: For text tokenization.

---

### Installation:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

### Conclusion:
This project provides a simple, easy-to-use **text analysis tool**, leveraging cutting-edge NLP models to generate titles, summaries, and detect emotions, making it valuable for content creation and analysis tasks.
