import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

CSV = "fr_hf.csv"
GRAINES = [42, 43, 44, 45, 46]
K = 15
COUT_FP, COUT_FN, COUT_H = 1, 5, 0.5
SEUILS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
SCENARIOS = [("Humain très bon marché", 1, 5, 0.1), ("Humain bon marché", 1, 5, 0.25), ("Nos valeurs", 1, 5, 0.5),
             ("Humain plus cher", 1, 5, 1), ("Humain très cher", 1, 5, 2),
             ("Message raté peu grave", 1, 2, 0.5), ("Message raté très grave", 1, 10, 0.5)]
SYSTEMES = ["Cascade", "Régression logistique + humain", "Régression logistique brute", "KNN brute"]


print("########## ETAPE 1 : donnees ##########")
dataset = pd.read_csv(CSV)
dataset = dataset[["text", "labels"]].dropna().drop_duplicates()
dataset = dataset.drop_duplicates(subset="text", keep=False)
texte = dataset["text"]
y = dataset["labels"].astype(int)


def confiance(p):
    return max(p, 1 - p)

def comptes(p1s, p2s, vrais, s1, s2):
    fp = fn = h = 0
    for p1, p2, v in zip(p1s, p2s, vrais):
        if confiance(p1) >= s1:
            d = int(p1 >= 0.5)
        elif confiance(p2) >= s2:
            d = int(p2 >= 0.5)
        else:
            h += 1
            continue
        if d == 1 and v == 0:
            fp += 1
        elif d == 0 and v == 1:
            fn += 1
    return fp, fn, h

def cout(c, n, a=COUT_FP, b=COUT_FN, h=COUT_H):
    return (a * c[0] + b * c[1] + h * c[2]) / n

def meilleurs_seuils(val, a, b, h, seuils2):
    meilleur, choix = 999, None
    for s1 in SEUILS:
        for s2 in seuils2:
            c = cout(val[(s1, s2)], val["n"], a, b, h)
            if c < meilleur:
                meilleur, choix = c, (s1, s2)
    return choix


print("########## ETAPE 2 : modeles et comptes pour chaque graine ##########")
donnees = []
for graine in GRAINES:
    print("graine", graine)
    t_reste, t_test, y_reste, y_test = train_test_split(texte, y, test_size=0.15, stratify=y, random_state=graine)
    t_train, t_val, y_train, y_val = train_test_split(
        t_reste, y_reste, test_size=0.15 / 0.85, stratify=y_reste, random_state=graine)
    tf_mots = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    tf_car = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X_train = hstack([tf_mots.fit_transform(t_train), tf_car.fit_transform(t_train)]).tocsr()
    X_val = hstack([tf_mots.transform(t_val), tf_car.transform(t_val)]).tocsr()
    X_test = hstack([tf_mots.transform(t_test), tf_car.transform(t_test)]).tocsr()
    modele_rl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    modele_knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", algorithm="brute").fit(X_train, y_train)

    p1_val = [p[1] for p in modele_rl.predict_proba(X_val)]
    p2_val = [p[1] for p in modele_knn.predict_proba(X_val)]
    p1_test = [p[1] for p in modele_rl.predict_proba(X_test)]
    p2_test = [p[1] for p in modele_knn.predict_proba(X_test)]
    v_val, v_test = list(y_val), list(y_test)

    val = {"n": len(v_val)}
    test = {"n": len(v_test)}
    for s1 in SEUILS:
        for s2 in SEUILS + [1.01]:
            val[(s1, s2)] = comptes(p1_val, p2_val, v_val, s1, s2)
            test[(s1, s2)] = comptes(p1_test, p2_test, v_test, s1, s2)
    test["knn"] = comptes(p1_test, p2_test, v_test, 1.01, 0.5)
    partage_rl = {s1: sum(confiance(p) >= s1 for p in p1_val) / len(p1_val) for s1 in SEUILS}
    donnees.append((val, test, partage_rl))


print("########## ETAPE 3 : sensibilite aux couts ##########")
lignes_tex = []
lignes_aff = []
for nom, a, b, h in SCENARIOS:
    couts = {s: [] for s in SYSTEMES}
    part_humain = []
    for val, test, _ in donnees:
        s1, s2 = meilleurs_seuils(val, a, b, h, SEUILS)
        s1_h, _ = meilleurs_seuils(val, a, b, h, [1.01])
        couts["Cascade"].append(cout(test[(s1, s2)], test["n"], a, b, h))
        couts["Régression logistique + humain"].append(cout(test[(s1_h, 1.01)], test["n"], a, b, h))
        couts["Régression logistique brute"].append(cout(test[(0.5, 0.5)], test["n"], a, b, h))
        couts["KNN brute"].append(cout(test["knn"], test["n"], a, b, h))
        part_humain.append(test[(s1, s2)][2] / test["n"])
    m = {s: sum(v) / len(v) for s, v in couts.items()}
    ph = sum(part_humain) / len(part_humain)
    lignes_aff.append((nom, a, b, h, m["Cascade"], m["Régression logistique + humain"],
                       m["Régression logistique brute"], m["KNN brute"], ph))
    lignes_tex.append(f"{nom} & {a} & {b} & {h} & {m['Cascade']:.3f} & {m['Régression logistique + humain']:.3f} & "
                      f"{m['Régression logistique brute']:.3f} & {m['KNN brute']:.3f} & {100 * ph:.1f}\\,\\% \\\\")
tableau = pd.DataFrame(lignes_aff, columns=["Scénario", "a", "b", "h", "Cascade", "RL + humain", "RL brute", "KNN brute", "Part humain"])
print(tableau.round(3).to_string(index=False))
with open("tableau_sensibilite.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lignes_tex) + "\n")


print("########## ETAPE 4 : figures ##########")
couts_systemes = {s: [] for s in SYSTEMES}
for val, test, _ in donnees:
    s1, s2 = meilleurs_seuils(val, COUT_FP, COUT_FN, COUT_H, SEUILS)
    s1_h, _ = meilleurs_seuils(val, COUT_FP, COUT_FN, COUT_H, [1.01])
    couts_systemes["Cascade"].append(cout(test[(s1, s2)], test["n"]))
    couts_systemes["Régression logistique + humain"].append(cout(test[(s1_h, 1.01)], test["n"]))
    couts_systemes["Régression logistique brute"].append(cout(test[(0.5, 0.5)], test["n"]))
    couts_systemes["KNN brute"].append(cout(test["knn"], test["n"]))
cout_df = pd.DataFrame(couts_systemes)
print(cout_df.agg(["mean", "std"]).round(3).to_string())

noms = ["Cascade", "RL + humain", "RL brute", "KNN brute"]
plt.figure(figsize=(6.5, 3.8))
plt.bar(noms, cout_df.mean(), yerr=cout_df.std(), capsize=4, color=["#2a9d8f", "#74c0b7", "#8d99ae", "#8d99ae"])
plt.ylabel("coût moyen par message")
plt.tight_layout()
plt.savefig("fig_cout.png", dpi=200)
plt.close()

grille = [[sum(cout(v[(s1, s2)], v["n"]) for v, _, _ in donnees) / len(donnees) for s2 in SEUILS] for s1 in SEUILS]
plt.figure(figsize=(6.5, 4.6))
plt.imshow(grille, cmap="RdYlGn_r", origin="lower")
plt.colorbar(label="coût moyen (validation)")
plt.xticks(range(len(SEUILS)), SEUILS, rotation=45)
plt.yticks(range(len(SEUILS)), SEUILS)
plt.xlabel("seuil2 (k-NN)")
plt.ylabel("seuil1 (régression logistique)")
i = min(range(len(SEUILS)), key=lambda r: min(grille[r]))
j = grille[i].index(min(grille[i]))
plt.scatter([j], [i], marker="s", s=140, facecolors="none", edgecolors="black", linewidths=2)
plt.tight_layout()
plt.savefig("fig_seuils.png", dpi=200)
plt.close()
print("minimum de la grille moyenne : seuil1 =", SEUILS[i], "| seuil2 =", SEUILS[j], "| cout =", round(min(grille[i]), 3))

part_rl = [sum(d[2][s1] for d in donnees) / len(donnees) for s1 in SEUILS]
part_h = [sum(d[0][(s1, 0.95)][2] / d[0]["n"] for d in donnees) / len(donnees) for s1 in SEUILS]
part_knn = [1 - a - b for a, b in zip(part_rl, part_h)]
plt.figure(figsize=(6.5, 3.8))
plt.bar([str(s) for s in SEUILS], part_rl, label="régression logistique", color="#2a9d8f")
plt.bar([str(s) for s in SEUILS], part_knn, bottom=part_rl, label="k-NN", color="#e9c46a")
plt.bar([str(s) for s in SEUILS], part_h, bottom=[a + b for a, b in zip(part_rl, part_knn)], label="humain", color="#e76f51")
plt.xlabel("seuil1 (seuil2 = 0,95)")
plt.ylabel("part des messages (validation)")
plt.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
plt.savefig("fig_decideurs.png", dpi=200)
plt.close()
print("a seuil1 = 0.75 : RL", round(part_rl[5], 3), "| k-NN", round(part_knn[5], 3), "| humain", round(part_h[5], 3))
