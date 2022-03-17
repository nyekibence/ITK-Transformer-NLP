# ITK-Transformer-NLP
Bevezetés az NLP-be Transformer-alapú modellekkel

## Tematika

Jelölések:

* (E) elméleti áttekintés
* (K) kód, implementáció bemutatása
* (E+K) elmélet és kód, elvi megoldások és implementáció
* (P) kidolgozott példa, a munkafolyamat bemutatása a bemenet előkészítésétől a kimenet értelmezéséig

A gyakorlatok a kidolgozott példákhoz hasonlóak, kiegészítendő kódot tartalmaznak, az önálló gyakorlást segítik.

### Ismerkedés a Transformer-alapú modellekkel
2022. március 7.
* Mi a Transformer architektúra és mire jó? (E)
* Az Attention mechanizmus (E+K)
* Attention súlyok kinyerése egy előtanított modellből (K)
* Betanított modell használata gépi fordításra (P)
* A Hugging Face transformers könyvtára (K)
* Gyakorlat: Question answering enkóder-dekóder modellel
 
### Enkóderek
2022. március 21.
* Transfer learning az NLP-ben: előnyök és hátrányok (E)
* A BERT modellek: az előtanítás (E+K)
* Finomhangolás szentimentanalízisre (P)
* Sentence-BERT tanítása NLI dataseten (E+K)
* Gyakorlat: Enkódermodell finomhangolása CoLA dataseten

### Dekóderek
2022. március 28.
* „Mi a következő token?”: A nyelvmodellezés elvei (E)
* Dekódolás Top-k random mintavételezéssel (E+K)
* Szöveggenerálás GPT-2 modellel (P)
* Az időre és memóriára optimalizált Sparse Transformer Attention mechanizmusa (E+K)
* Gyakorlat: Összefoglalógenerálás dekódermodellel
 
### Haladó témák
2022. április 4.
* Hiperparaméter-hangolás (E+K)
* Knowledge distillation (E+K)
* Cross-lingual few-shot és zero-shot (E)
* Cross-lingual few-shot osztályozás (P)
* Gyakorlat: Multilingual knowledge distillation

## Telepítés, használat
Klónozzuk a fájlrendszert:

```bash
git clone https://github.com/nyekibence/ITK-Transformer-NLP.git
```

Az előadásokon bemutatott notebook-ok a `matrials` könyvtárban érhetők el, a gyakorlatok pedig
az `itk_transformer_nlp`-ben. Az utóbbi Python csomagba van szervezve a használat és a tesztelés megkönnyítése végett.

Telepítsük a [poetry](https://python-poetry.org/docs/#installation) eszközt, hozzunk létre új virtuális környezetet
a Python 3.9-es verziójával, majd futtassuk a következő parancsot:

```bash
poetry install && python3 setup.py develop
```

## Gyakorlatok
A gyakorlatok kiegészítendő kódok. A függvénytestekben a `None` objektumokat kell saját kódra cserélni. Részletesebb instrukciók a kódfájlokban.

A futtatáshoz és teszeteléshez elérhetők `make` parancsok. 

### Question answering
Question answering a [bart-squadv2](https://huggingface.co/a-ware/bart-squadv2) segítségével. Ez egy nagyméretű modell,
a gyakorlatban való használata előtt ajánlott, hogy legalább 8 GB RAM legyen elérhető. 

Szkript: `itk_transformer_nlp/transformer_qa.py`

Make parancsok:
* `data/squad_example.jsonl`: Letölt egy rövid mintát a [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) datasetből `jsonlines` formátumban.
* `qa_on_squad`: Futtatja a teszteket és kiírja a letöltött SQuAD adatpontokon kiszámolt predikciókat. Ezzel lehet ellenőrizni, helyesek-e a gyakorlat megoldásai.
* `qa_solutions:` Beírja a megoldásokat a `itk_transformer_nlp/transformer_qa.py` fájlba.
* `qa_reload_lab`: Visszaállítja az eredeti kiegészítendő kódot az `itk_transformer_nlp/transformer_qa.py` fájlban.