from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import numpy as np
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download("stopwords")
stop_words = stopwords.words("english")
modelo_carregado = load_model('my_model.keras')

def lemmatize_words(tokenized_word):
    return lemmatizer.lemmatize(tokenized_word)

def Preprocessamento(text):  # tokenização, casefolding, lematização
    result = []
    words = tokenizer.tokenize(text)
    for word in words:
        word = word.casefold() 
        if word not in stop_words:   
            word = lemmatize_words(word)  
            result.append(word)  
    return result

fake_news = [
    "Donald Trump has announced plans to build a wall around the entire United States, stating that it will be funded by a special tax on all American citizens.",
    "Former President Trump has been discovered to have a hidden offshore bank account used exclusively for funneling donations from foreign lobbyists.",
    "A leaked document reveals that Trump has been in communication with extraterrestrial beings, planning to unveil a new intergalactic policy during his next campaign.",
    "A prominent scientist has claimed that drinking three cups of coffee a day can grant you superhuman intelligence and enhance your cognitive abilities dramatically.",
    "Reports have emerged that a group of scientists has discovered a way to resurrect dinosaurs using advanced genetic engineering techniques, with plans to open a theme park next year.",
    "A viral video claims that a famous celebrity has turned into a vampire after visiting a secret society that promises immortality to its members.",
    "A so-called whistleblower has alleged that a major tech company is secretly monitoring users through their smart refrigerators to gather personal data for political campaigns.",
    "An article suggests that a popular children's cartoon character is actually a coded message encouraging children to join a secret cult."
]

# Lista de True News
true_news = [
    "Donald Trump has announced his candidacy for the 2024 presidential election, stating his intention to run and address issues such as the economy and immigration.",
    "Former President Trump is facing multiple legal challenges, including investigations into his business practices and the January 6 Capitol riots.",
    "In a recent speech, Trump criticized the Biden administration's handling of the economy, emphasizing his own policies during his presidency that he claims led to economic growth.",
    "Donald Trump recently spoke at a rally where he outlined his key policies and addressed his supporters about his vision for America.",
    "Former President Trump has announced a series of public events to discuss his views on health care reform and national security.",
    "In an interview, Trump stated his commitment to addressing climate change through innovation and technology during his next term.",
    "Recent studies show that unemployment rates have dropped significantly since the implementation of new economic policies in various states.",
    "Experts are analyzing the impact of Trump's trade policies on the agricultural sector, highlighting both challenges and opportunities for farmers."
]

def prever_noticia(text):

    text = Preprocessamento(text)
    

    text_ = " ".join(text)
    

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer1 = pickle.load(handle)
    

    novos_textos_seq = tokenizer1.texts_to_sequences([text_])
    novos_textos_seq = [seq for seq in novos_textos_seq if seq]  # Manter apenas sequências não vazias
    
    if not novos_textos_seq:
        raise ValueError("erro")
    

    novos_textos_pad = pad_sequences(novos_textos_seq, maxlen=100)
    
    # Fazer a previsão
    pred = modelo_carregado.predict(novos_textos_pad)
    
    # Retornar a classe prevista
    return pred

for news in fake_news:

    pred = prever_noticia(news)
    # 1 para noticia falsa e 0 para noticia verdadeira
    if(pred >= 0.6 ):
        print("noticia falsa")
        print(pred)
    else:
        print("noticia verdadeira")

print("---------------------------------------")
for news in true_news:

    pred = prever_noticia(news)
    # 1 para noticia falsa e 0 para noticia verdadeira
    if(pred >= 0.6 ):
        print("noticia falsa")
        print(pred)
    else:
        print("noticia verdadeira")
        print(pred)

