from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def quitarStopwords_eng(texto):
    ingles = stopwords.words("english")
    texto_limpio = [w.lower() for w in texto if w.lower() not in ingles 
                    and w not in string.punctuation 
                    and w not in ["'s", '|', '--', "''", "``"] ]
    return texto_limpio

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    texto_lema = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]
    return texto_lema

#Inicializar el Lematizador
lemmatizer = WordNetLemmatizer()

#Definir el corpus de texto
corpus = [
   lematizar(quitarStopwords_eng(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]

corpus_final = []

for oracion in corpus:
    resultado = ' '.join(oracion) 
    corpus_final.append(resultado)

print(corpus_final)

#Inicializar el TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_final)
print("Matriz TF-IDF:")
print(X.toarray())
print("\nVocabulario:")
print(vectorizer.get_feature_names_out())



from collections import Counter
import numpy as np

# Unir todo el corpus en una sola lista de palabras
todas_palabras = []
for oracion in corpus:
    todas_palabras.extend(oracion)

# Contar frecuencia
frecuencia = Counter(todas_palabras)

print("\nFrecuencia de palabras:")
print(frecuencia)


# 1. Top 6 palabras más usadas
top6 = frecuencia.most_common(6)
print("\nTop 6 palabras más usadas:")
print(top6)


# 2. Palabra menos utilizada
menos_usada = min(frecuencia, key=frecuencia.get)
print("\nPalabra menos utilizada:")
print(menos_usada, frecuencia[menos_usada])


# 3. Palabras más repetidas en la misma oración
print("\nPalabras más repetidas por oración:")
for i, oracion in enumerate(corpus):
    freq_oracion = Counter(oracion)
    mas_comun = freq_oracion.most_common(1)
    print(f"Oración {i+1}: {mas_comun}")


# 4. Gráfico de distribución de frecuencia
import matplotlib.pyplot as plt

palabras = list(frecuencia.keys())
valores = list(frecuencia.values())

plt.figure()
plt.bar(palabras, valores)
plt.xticks(rotation=90)
plt.title("Distribución de Frecuencia de Palabras")
plt.show()