import os
import pandas as pd
import json
from pandas.io.json import json_normalize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from copy import deepcopy

"""
DescToGenre mudeli genereerimise kood.
"""

def read_texts(folder_name):
    """
    Loeb filmi JSON failid ja tagastab tabeli, kus üks veerg tähistab (ühte) filmi žanri, teine filmi kirjeldust
    :param folder_name: kausta nimi, kus filmi JSON'id on
    :return: Tagastab tabeli
    """
    datalist = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            with open(os.path.join(root, file), encoding="utf8") as f:
                df = json.load(f)
                if len(df['movies']) == 0:
                    continue
            df = pd.DataFrame(json_normalize(df["movies"]))
            df = df.drop(columns=["id", "title"])

            for index, row in df.iterrows():
                if len(row['overview']) == 0 or len(row['overview'].split()) < 10:
                    continue
                for genre in json.loads(str(row['genres']).replace("'", "\"")):
                    if genre["name"] not in ["TV Movie", "Foreign"]:
                        datalist.append(pd.DataFrame({'genre': [genre["name"]], 'overview': row["overview"]}))

    genres = pd.concat(datalist)
    return genres.reset_index(drop=True)

def concat_genres(data):
    """
    Loob olemasolevast kirjelduste ja žanrite tabelist uue tabeli, kus filmile vastavad kõik žanrid, mis neile määratud
    :param data: DataFrame, kus esimene veerg on filmi žanr ja teine filmi kirjeldus. Ühe filmi kirjed järjest
    :return: Uus tabel, kus esimene veerg on filmi kõik žanrid ja teine filmi kirjeldus
    """
    combined_genre = []
    previous = ""
    for index, row in data.iterrows():
        if previous != row["overview"]:
            combined_genre.append(deepcopy(row))
            combined_genre[-1]["genre"] = [combined_genre[-1]["genre"]]
            previous = row["overview"]
        else:
            current = combined_genre[-1]["genre"]
            if type(current) != list:
                combined_genre[-1]["genre"] = [current]
                combined_genre[-1]["genre"].append(row["genre"])
            else:
                combined_genre[-1]["genre"].append(row["genre"])

    return pd.DataFrame(combined_genre).reset_index(drop=True)


# Lähteandmed
print("loading data")
data = read_texts('movies')
combined_genre = concat_genres(data)

# Eeltöötlus - teeme filmidest TaggedDocument objektid
print("preprocessing data")
documents = []
documentsWithCombinedGenres = []

for i in range(len(data)):
    documents.append(TaggedDocument(data["overview"][i].split(), [data["genre"][i]]))
    if i < len(combined_genre):
        documentsWithCombinedGenres.append(TaggedDocument(combined_genre["overview"][i].split(), [combined_genre["genre"][i]]))


print("training model")
model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
#model.save("DescToGenre")                      Mudeli salvestamine
#model = Doc2Vec.load("DescToGenre")            Mudeli laadimine

# model = Doc2Vec(documents[100:], size=100, window=8, min_count=5, workers=4) <- Kasutada testimisel

# Testimine - võtame esimesed 100 filmi ja vaatame, kas neile ennustatakse vähemalt üks žanr õigesti
print("testing")
result = 0
for document in documentsWithCombinedGenres[:100]:
    inferred_docvec = model.infer_vector(document.words)
    prediction = model.docvecs.most_similar([inferred_docvec], topn=3)
    for predicted in prediction:
        if predicted[0] in document[1][0]:
            result += 1
            break
    #print("correct:", document[1], "prediction:", prediction)


print(result / 100)