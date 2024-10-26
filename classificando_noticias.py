import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Exemplo de URL de notícia: https://edition.cnn.com/2022/11/15/politics/trump-2024-presidential-bid/index.html
# Exemplo de fake news por texto: "Donald Trump has announced plans to build a wall around the entire United States, stating that it will be funded by a special tax on all American citizens."

# Download de dados do NLTK
nltk.download('wordnet')
nltk.download("stopwords")

# Inicialização de ferramentas
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Carregar o tokenizer salvo
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

# Função para pré-processamento de texto
def preprocess_text(text):
    words = tokenizer.tokenize(text)
    processed_words = [
        lemmatizer.lemmatize(word.casefold())
        for word in words if word.casefold() not in stop_words
    ]
    return processed_words

# Função de previsão
def predict_news(text, model):
    processed_text = preprocess_text(text)
    joined_text = " ".join(processed_text)
    
    # Converter texto para sequência
    text_sequence = tokenizer1.texts_to_sequences([joined_text])
    text_sequence = [seq for seq in text_sequence if seq]  # Manter sequências não vazias
    
    if not text_sequence:
        raise ValueError("Erro: A sequência de texto processada está vazia após a tokenização.")

    # Preencher a sequência para ter o tamanho necessário
    padded_sequence = pad_sequences(text_sequence, maxlen=100)
    
    # Fazer a previsão
    prediction = model.predict(padded_sequence)
    return prediction

# Função para obter notícia a partir de um URL
def fetch_news_from_url(link):
    try:
        response = requests.get(link)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').get_text().strip() if soup.find('h1') else "Título não encontrado"
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        return f"{title}\n\n{content}"
    except Exception as e:
        return f"Ocorreu um erro: {str(e)}"

# Função principal para solicitar escolha de entrada e de modelo ao usuário
def main():
    # Escolha do modelo pelo usuário
    model_choice = input("Escolha o modelo de predição:\n1 - LSTM\n2 - CNN\nDigite a escolha (1 ou 2): ")
    
    if model_choice == '1':
        model = load_model('modelo_LSTM.keras')
        print("Modelo LSTM carregado com sucesso.")
    elif model_choice == '2':
        model = load_model('modelo_CNN.keras')
        print("Modelo CNN carregado com sucesso.")
    else:
        print("Escolha inválida. Por favor, digite 1 ou 2.")
        return

    # Escolha do método de entrada
    input_choice = input("Escolha o método de entrada:\n1 - URL\n2 - Texto direto\nDigite a escolha (1 ou 2): ")

    if input_choice == '1':
        link = input("Digite o URL do artigo de notícia: ")
        news_text = fetch_news_from_url(link)
    elif input_choice == '2':
        news_text = input("Digite o texto da notícia: ")
    else:
        print("Escolha inválida. Por favor, digite 1 ou 2.")
        return

    # Prever e exibir resultado
    try:
        prediction = predict_news(news_text, model)
        print("Previsão {Proximo de 0 Para noticia real|e Proximo de 1 para fakenews|} :", prediction)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
