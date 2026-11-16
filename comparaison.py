import time
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

CSV = "C:/fr_hf.csv"
GRAINES = [42, 43, 44, 45, 46]
K = 15
COUT_FP, COUT_FN, COUT_H = 1, 5, 0.5
SEUILS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
AUTRES_MODELES = {}


# pretraitement
print("########## ETAPE 1 : collecte ##########")
dataset = pd.read_csv(CSV)
print(dataset.shape)

print("########## ETAPE 2 : nettoyage ##########")
dataset = dataset[["text", "labels"]]
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
dataset = dataset.drop_duplicates(subset="text", keep=False)
print(dataset.shape)

print("########## ETAPE 3 : x et y ##########")
texte = dataset["text"]
y = dataset["labels"].astype(int)
print("part de messages nuisibles :", round(y.mean(), 2))


def confiance(p):
    return max(p, 1 - p)

def cascade(p1, p2, seuil1, seuil2):
    if confiance(p1) >= seuil1:
        return int(p1 >= 0.5)
    if confiance(p2) >= seuil2:
        return int(p2 >= 0.5)
    return None

def cout_moyen(decisions, vrais):
    total = 0
    for d, v in zip(decisions, vrais):
        if d is None:
            total += COUT_H
        elif d == 1 and v == 0:
            total += COUT_FP
        elif d == 0 and v == 1:
            total += COUT_FN
    return total / len(vrais)

def chercher_seuils(p1s, p2s, vrais, seuils2=SEUILS):
    meilleur_cout, seuil1, seuil2 = 999, 0.5, 0.5
    for s1 in SEUILS:
        for s2 in seuils2:
            decisions = [cascade(a, b, s1, s2) for a, b in zip(p1s, p2s)]
            c = cout_moyen(decisions, vrais)
            if c < meilleur_cout:
                meilleur_cout, seuil1, seuil2 = c, s1, s2
    return seuil1, seuil2

def mesures(decisions, vrais):
    repondus = [(d, v) for d, v in zip(decisions, vrais) if d is not None]
    d_rep = [d for d, v in repondus]
    v_rep = [v for d, v in repondus]
    bonnes = sum(1 for d, v in repondus if d == v)
    faux_pos = sum(1 for d, v in repondus if d == 1 and v == 0)
    negatifs = sum(1 for d, v in repondus if v == 0)
    return {"Exactitude": bonnes / len(repondus),
            "Précision": precision_score(v_rep, d_rep, zero_division=0),
            "Rappel": recall_score(v_rep, d_rep, zero_division=0),
            "F1 macro": f1_score(v_rep, d_rep, average="macro"),
            "Faux positifs": faux_pos / negatifs,
            "Envoyé à l'humain": decisions.count(None) / len(decisions),
            "Coût": cout_moyen(decisions, vrais)}


lignes = []
for graine in GRAINES:
    print(f"\n########## GRAINE {graine} ##########")

    print("ETAPE 4 : decoupage")
    t_reste, t_test, y_reste, y_test = train_test_split(
        texte, y, test_size=0.15, stratify=y, random_state=graine)
    t_train, t_val, y_train, y_val = train_test_split(
        t_reste, y_reste, test_size=0.15 / 0.85, stratify=y_reste, random_state=graine)

    # vectorisation
    print("ETAPE 5 : TF-IDF")
    tf_mots = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    tf_car = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X_train = hstack([tf_mots.fit_transform(t_train), tf_car.fit_transform(t_train)]).tocsr()
    X_val = hstack([tf_mots.transform(t_val), tf_car.transform(t_val)]).tocsr()
    X_test = hstack([tf_mots.transform(t_test), tf_car.transform(t_test)]).tocsr()

    # entrainement
    print("ETAPE 6 : entrainement")
    modele_rl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    modele_knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", algorithm="brute").fit(X_train, y_train)

    # seuils
    print("ETAPE 7 : choix des seuils")
    p1_val = [p[1] for p in modele_rl.predict_proba(X_val)]
    p2_val = [p[1] for p in modele_knn.predict_proba(X_val)]
    vrais_val = list(y_val)
    seuil1, seuil2 = chercher_seuils(p1_val, p2_val, vrais_val)
    seuil_h, _ = chercher_seuils(p1_val, p2_val, vrais_val, [1.01])
    print(f"   seuil1 = {seuil1} | seuil2 = {seuil2}")

    # test
    print("ETAPE 8 : test")
    vrais = list(y_test)

    debut = time.perf_counter()
    p1_test = [p[1] for p in modele_rl.predict_proba(X_test)]
    temps_rl = time.perf_counter() - debut
    dec_rl = [int(p >= 0.5) for p in p1_test]
    dec_rl_humain = [cascade(p1, 0.5, seuil_h, 1.01) for p1 in p1_test]

    debut = time.perf_counter()
    p2_test = [p[1] for p in modele_knn.predict_proba(X_test)]
    temps_knn = time.perf_counter() - debut
    dec_knn = [int(p >= 0.5) for p in p2_test]

    # cascade
    douteux = [i for i, p in enumerate(p1_test) if confiance(p) < seuil1]
    p2_douteux = {}
    debut = time.perf_counter()
    if douteux:
        for i, p in zip(douteux, modele_knn.predict_proba(X_test[douteux])):
            p2_douteux[i] = p[1]
    temps_cascade = temps_rl + (time.perf_counter() - debut)
    dec_cascade = [cascade(p1, p2_douteux.get(i, 0.5), seuil1, seuil2) for i, p1 in enumerate(p1_test)]

    systemes = {"Cascade": (dec_cascade, temps_cascade),
                "Régression logistique + humain": (dec_rl_humain, temps_rl),
                "Régression logistique brute": (dec_rl, temps_rl),
                "KNN brute": (dec_knn, temps_knn)}

    for nom, modele in AUTRES_MODELES.items():
        modele.fit(X_train, y_train)
        debut = time.perf_counter()
        dec = [int(d) for d in modele.predict(X_test)]
        systemes[nom] = (dec, time.perf_counter() - debut)

    for nom, (decisions, duree) in systemes.items():
        ligne = mesures(decisions, vrais)
        ligne["Temps (ms/msg)"] = 1000 * duree / len(vrais)
        ligne["Système"] = nom
        ligne["Graine"] = graine
        lignes.append(ligne)


# resultats
print("\n########## ETAPE 9 : resultats ##########")
tableau = pd.DataFrame(lignes)
tableaux = [["Exactitude", "Précision", "Rappel", "F1 macro"],
            ["Faux positifs", "Envoyé à l'humain", "Coût", "Temps (ms/msg)"]]
colonnes = tableaux[0] + tableaux[1]
pourcentages = ["Exactitude", "Précision", "Rappel", "F1 macro", "Faux positifs", "Envoyé à l'humain"]
moyenne = tableau.groupby("Système", sort=False)[colonnes].mean()
ecart = tableau.groupby("Système", sort=False)[colonnes].std()

affichage = pd.DataFrame(index=moyenne.index)
for col in colonnes:
    if col in pourcentages:
        affichage[col] = [f"{100 * m:.1f} % ± {100 * e:.1f}" for m, e in zip(moyenne[col], ecart[col])]
    elif col == "Coût":
        affichage[col] = [f"{m:.3f} ± {e:.3f}" for m, e in zip(moyenne[col], ecart[col])]
    else:
        affichage[col] = [f"{m:.4f} ± {e:.4f}" for m, e in zip(moyenne[col], ecart[col])]
for cols in tableaux:
    print(affichage[cols].to_string())
    print()

with open("tableau_resultats.tex", "w", encoding="utf-8") as f:
    for i, cols in enumerate(tableaux, start=1):
        f.write(f"% tableau {i}\n")
        for nom in affichage.index:
            cases = [affichage.loc[nom, c].replace("%", "\\%").replace("±", "$\\pm$") for c in cols]
            f.write(nom + " & " + " & ".join(cases) + " \\\\\n")
