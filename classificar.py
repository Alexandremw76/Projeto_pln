from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import requests
from bs4 import BeautifulSoup
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

fake_news_política = [
    "Donald Trump has announced plans to build a wall around the entire United States, stating that it will be funded by a special tax on all American citizens.",
    "Former President Trump has been discovered to have a hidden offshore bank account used exclusively for funneling donations from foreign lobbyists.",
    "A leaked document reveals that Trump has been in communication with extraterrestrial beings, planning to unveil a new intergalactic policy during his next campaign.",
    "A prominent scientist has claimed that drinking three cups of coffee a day can grant you superhuman intelligence and enhance your cognitive abilities dramatically.",
    "Reports have emerged that a group of scientists has discovered a way to resurrect dinosaurs using advanced genetic engineering techniques, with plans to open a theme park next year.",
    "A viral video claims that a famous celebrity has turned into a vampire after visiting a secret society that promises immortality to its members.",
    "An article suggests that a popular children's cartoon character is actually a coded message encouraging children to join a secret cult."
]

true_news_política = [
    "Donald Trump has announced his candidacy for the 2024 presidential election, stating his intention to run and address issues such as the economy and immigration.",
    "Former President Trump is facing multiple legal challenges, including investigations into his business practices and the January 6 Capitol riots.",
    "In a recent speech, Trump criticized the Biden administration's handling of the economy, emphasizing his own policies during his presidency that he claims led to economic growth.",
    "Donald Trump recently spoke at a rally where he outlined his key policies and addressed his supporters about his vision for America.",
    "Former President Trump has announced a series of public events to discuss his views on health care reform and national security.",
    "In an interview, Trump stated his commitment to addressing climate change through innovation and technology during his next term.",
    "Recent studies show that unemployment rates have dropped significantly since the implementation of new economic policies in various states.",
    "Experts are analyzing the impact of Trump's trade policies on the agricultural sector, highlighting both challenges and opportunities for farmers."
]

fake_news_articles = [
    "Scientists Discover Cure for All Diseases Using Only Water: A groundbreaking study claims that drinking purified water can cure chronic and infectious diseases. Researchers suggest that hydration alone can eliminate the need for medical treatment, ignoring the substantial evidence that proper medical care is essential for disease management.",
    
    "New Research Reveals Eating Chocolate Makes You Live Longer: An article claims that daily chocolate consumption significantly increases life expectancy. However, it overlooks the health risks associated with excessive sugar and fat intake, promoting an unrealistic and unhealthy lifestyle.",
    
    "Scientific Experiment Proves Vaccines Cause Autism: A widely discredited, unreviewed study is used to support claims that vaccines are directly linked to the rise in autism cases. This claim disregards decades of rigorous research that demonstrate the safety and efficacy of vaccines.",
    
    "Newly Discovered Planet Can Sustain Human Life Without Any Changes: Astronomers announce the discovery of a new planet that supposedly can support human life without any modifications. The claim ignores the complexities of planetary ecosystems and the fundamental requirements for human survival.",
    
    "Study Claims Listening to Music Improves Intelligence: A recent study suggests that listening to classical music can enhance cognitive abilities and increase intelligence. However, the study fails to control for other variables and presents anecdotal evidence rather than scientific proof."
]

real_news_articles = [
    "NASA's Perseverance Rover Finds Signs of Ancient Life on Mars: NASA's Perseverance rover has discovered organic molecules and seasonal changes in methane on Mars, suggesting that the planet may have supported ancient microbial life. This finding adds to the growing body of evidence that Mars was once a habitable environment.",
    "Scientists Create Functional Human Organs from Stem Cells: Researchers have successfully grown functional human organs using stem cells in a laboratory setting. This breakthrough could lead to significant advancements in transplantation medicine, potentially addressing the shortage of organ donors.",
    "Climate Change Accelerates Global Warming: A recent report from the Intergovernmental Panel on Climate Change (IPCC) warns that global temperatures could rise by 1.5 degrees Celsius as early as 2030. This increase poses serious risks to ecosystems and human life, emphasizing the urgent need for climate action.",
    "CRISPR Technology Shows Promise in Treating Genetic Disorders: Scientists are using CRISPR gene-editing technology to target and potentially cure genetic disorders like sickle cell anemia and muscular dystrophy. These developments could revolutionize the treatment of hereditary diseases.",
    "Discovery of a New Exoplanet in the Habitable Zone: Astronomers have identified a new exoplanet located in the habitable zone of its star, which could potentially support liquid water. This discovery raises exciting possibilities for the search for extraterrestrial life."
]
elaborate_fake_news_articles = [
    "Breakthrough Study Reveals Secret Ingredient in Common Foods Prevents Cancer: A recent study conducted by a team of scientists claims that a compound found in everyday foods, such as broccoli and spinach, has been proven to prevent cancer cell growth. However, the research lacks peer review and has not been replicated in controlled trials, raising questions about its validity.",
    
    "New Vaccine Linked to Unexplained Increase in Fertility Issues: An independent research group has published findings suggesting a correlation between a newly released vaccine and an increase in fertility problems among women in specific regions. While the study appears to provide data supporting its claims, it fails to consider other variables and has not been substantiated by larger studies.",
    
    "Ancient Manuscript Claims Humans Can Live for Hundreds of Years: A group of historians has uncovered an ancient manuscript that suggests humans once had the potential to live for several centuries. The claims are based on historical interpretations rather than solid archaeological evidence, leading to sensationalist headlines that misrepresent the actual findings.",
    
    "Scientists Announce Discovery of 'Anti-Gravity' Technology: A team of physicists has allegedly developed a technology that can negate the effects of gravity, claiming it will revolutionize transportation. However, the publication is from a questionable source, and the scientific community has expressed skepticism regarding the feasibility of such technology.",
    
    "Groundbreaking Research Links Smartphone Use to Increased IQ: A new study suggests that regular use of smartphones enhances cognitive abilities and can lead to an increase in IQ scores. Despite the seemingly supportive data, the study has not undergone rigorous scientific evaluation, and the claims are based on self-reported measures rather than standardized tests."
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
    return print(pred)

def pegar_noticia_url(link):
    try:
        response = requests.get(link)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('h1').get_text().strip() if soup.find('h1') else "No title found"
        
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs])

 
        full_text = f"{title}\n\n{text}"
        
        return full_text
    except Exception as e:
        return f"Error occurred: {str(e)}"


url = ""

prever_noticia(real_news_articles[0])

prever_noticia(real_news_articles[1])

prever_noticia(real_news_articles[2])

prever_noticia(real_news_articles[3])
prever_noticia(real_news_articles[4])