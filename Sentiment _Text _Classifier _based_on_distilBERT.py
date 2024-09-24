#Sentiment Text Classifier based on distilbert model 

import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    title_generator = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    return summarizer, emotion_classifier, title_generator

summarizer, emotion_classifier, title_generator = load_models()

def generate_title(text):
    """
    Function to generate a catchy title from the provided text using the T5 model.
    The title will be limited to a maximum of 10 words with no punctuation (except a full stop).
    """
    # Generate title using the summarization model
    summary = title_generator(text, max_length=15, min_length=5, do_sample=False)
    
    # Extract the generated title and clean it up
    title = summary[0]['summary_text'].strip()
    title_words = title.split()[:10]  # Limit to 10 words
    title = ' '.join(title_words).lower()  # Lowercase and remove other punctuation
    
    # Ensure there's a full stop at the end
    if not title.endswith('.'):
        title += '.'
    
    return title


def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def classify_emotion(text):
    results = emotion_classifier(text)
    emotions = [(item['label'], item['score'], get_emoji(item['label'])) for item in results[0]]
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions[:3]  # Return top 3 emotions

def get_emoji(emotion):
    emoji_map = {
        "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
        "desire": "😍", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
        "embarrassment": "😳", "excitement": "🎉", "fear": "😨", "gratitude": "🙏",
        "joy": "😄", "love": "❤️", "nervousness": "😰", "optimism": "🌟",
        "pride": "😊", "realization": "💡", "relief": "😌", "remorse": "😔",
        "sadness": "😢", "surprise": "😲", "neutral": "😐"
    }
    return emoji_map.get(emotion.lower(), "")

st.title("Text Analyzer")

text_input = st.text_area("Enter your text here:")

if st.button("Analyze Text"):
    if text_input:
        with st.spinner("Analyzing..."):
            title = generate_title(text_input)
            summary = summarize_text(text_input)
            emotions = classify_emotion(text_input)
        
        st.subheader("Generated Title:")
        st.write(title)
        
        st.subheader("Summary:")
        st.write(summary)
        
        st.subheader("Emotion Analysis Result:")
        for emotion, score, emoji in emotions:
            st.write(f"{emoji} {emotion.capitalize()}: {score:.2%}")
    else:
        st.warning("Please enter some text to analyze.")



#!wget -q -O - https://loca.lt/mytunnelpassword

#!streamlit run app.py & npx localtunnel --port 8501