import pandas as pd
from collections import Counter
import nltk
import re


def ucitaj_podatke(file_path):
    return pd.read_csv(file_path, sep='\t')

# 1. Distribucija sentimenta
def distribucija_sentimenta(df):
    sentiment_counts = df['Label'].value_counts()
    sentiment_labels = {0: 'Pozitivan', 1: 'Neutralan', 2: 'Negativan', 3: 'Mixed', 4: 'Sarkazam/Ironija'}
    sentiment_counts = sentiment_counts.rename(sentiment_labels)
    
    print("\nDistribucija sentimenta:")
    print(sentiment_counts)

# 2. Analiza duljine teksta (prosjecna duljina recenica i distribucija)
def analiza_duljine_teksta(df):
    df['Duljina_Recenice'] = df['Sentence'].apply(lambda x: len(x.split()))
    prosjecna_duljina = df['Duljina_Recenice'].mean()
    
    print(f"Prosjecna duljina recenica: {prosjecna_duljina:.2f} rijeci")
    
    print("\nDistribucija duljine recenica:")
    print(df['Duljina_Recenice'].describe())

# 3. Analiza najcesce koristenih rijeci
def analiza_najcescih_rijeci(df):
    svi_tekstovi = ' '.join(df['Sentence']).lower()
    rijeci = nltk.word_tokenize(svi_tekstovi)
    rijeci = [rijec for rijec in rijeci if rijec.isalnum()]  # uklanjanje interpunkcija
    
    brojac_rijeci = Counter(rijeci)
    najcesce_rijeci = brojac_rijeci.most_common(10)
    
    print("\nNajcesce koristene rijeci:")
    for rijec, broj in najcesce_rijeci:
        print(f"{rijec}: {broj}")

# 4. Analiza kompleksnosti recenica
def analiza_kompleksnosti_recenica(df):
    df['Broj_Rijeci'] = df['Sentence'].apply(lambda x: len(x.split()))
    df['Broj_Slogova'] = df['Sentence'].apply(lambda x: sum([len(word) for word in x.split() if word.isalnum()]))
    
    prosjecan_broj_rijeci = df['Broj_Rijeci'].mean()
    prosjecan_broj_slogova = df['Broj_Slogova'].mean()
    
    print(f"Prosjecan broj rijeci po recenici: {prosjecan_broj_rijeci:.2f}")
    print(f"Prosjecan broj slogova po recenici: {prosjecan_broj_slogova:.2f}")

# 5. Detekcija negacija
def detekcija_negacija(df):
    negacije = ['ne', 'nikad', 'nije', 'nema', 'nismo', 'neću', 'neće', 'nikako']
    
    # Detekcija negacija
    df['Sadrzi_Negaciju'] = df['Sentence'].apply(lambda x: any(negacija in x.lower() for negacija in negacije))
    
    negacija_sentiment_count = df.groupby('Sadrzi_Negaciju')['Label'].value_counts().unstack().fillna(0)
    print("\nRaspodjela sentimenta prema prisutnosti negacija:")
    print(negacija_sentiment_count)

def izvrsi_eda(file_path):
    df = ucitaj_podatke(file_path)
    distribucija_sentimenta(df)
    analiza_duljine_teksta(df)
    analiza_najcescih_rijeci(df)
    analiza_kompleksnosti_recenica(df)
    detekcija_negacija(df)

izvrsi_eda('OPJ_final_2.tsv')
