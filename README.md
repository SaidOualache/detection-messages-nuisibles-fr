# détection de messages nuisibles en français


Un modèle rapide (régression logistique) répond quand il est sûr de lui. Sinon, un second modèle (k-NN) donne son avis. Si le doute persiste, le message est envoyé à un modérateur humain.

Les deux seuils de confiance sont choisis sur le jeu de validation pour minimiser un coût moyen : bloquer à tort un message inoffensif = 1, laisser passer un message nuisible = 5, envoyer à un humain = 0,5. Ces trois valeurs sont des hypothèses, modifiables dans `comparaison.py`.

## Données

French Hate Speech Superset (Tonneau et al., 2024) :
https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset

L'accès est restreint. Demandez l'accès sur Hugging Face, téléchargez le CSV et placez-le à côté de `comparaison.py` sous le nom `fr_hf.csv`. Le fichier n'est pas inclus dans ce dépôt.

## Utilisation

```
pip install -r requirements.txt
python comparaison.py
```

Le script nettoie les données, les découpe en 70 % / 15 % / 15 % (graines 42 à 46), compare les systèmes sur le jeu de test et écrit `tableau_resultats.tex`.
Testé avec Python 3.12, pandas 3.0 et scikit-learn 1.9.

## Systèmes comparés

- **Cascade** : régression logistique, puis k-NN, puis humain
- **Régression logistique + humain** : la cascade sans le k-NN
- **Régression logistique brute**
- **KNN brute**

## Résultats

Jeu de test, moyenne ± écart-type sur 5 exécutions.

| Système | Exactitude | F1 macro | Faux positifs | Envoyé à l'humain | Coût |
|---|---|---|---|---|---|
| Cascade | 90,0 % ± 2,7 | 85,0 % ± 2,7 | 7,8 % ± 2,8 | 40,0 % ± 11,2 | 0,356 ± 0,014 |
| Régression logistique + humain | 90,3 % ± 2,4 | 85,6 % ± 2,5 | 7,8 % ± 2,4 | 44,0 % ± 9,3 | 0,355 ± 0,014 |
| Régression logistique brute | 80,0 % ± 1,0 | 74,9 % ± 1,2 | 17,7 % ± 0,6 | 0,0 % | 0,462 ± 0,031 |
| KNN brute | 79,6 % ± 0,6 | 62,5 % ± 1,4 | 3,1 % ± 0,4 | 0,0 % | 0,923 ± 0,025 |

Pour les systèmes avec humain, l'exactitude, le F1 et les faux positifs ne comptent que les messages traités par le modèle. Le coût est la comparaison la plus juste : plus il est bas, mieux c'est.

La cascade et « régression logistique + humain » ont le même coût : le gain vient de la zone de révision humaine, pas du k-NN.

