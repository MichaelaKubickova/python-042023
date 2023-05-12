import pandas as pd
import numpy
import matplotlib.pyplot as plt
data = pd.read_csv("python-042023/Ukol1/1976-2020-president.csv")
#print(data.dtypes)

#1.cast ukolu
#urči pořadí jednotlivých kandidátů v jednotlivých státech a v jednotlivých letech pomocí metody rank()
#data = data.sort_values(["state"])
data["rank"] = data.groupby(["year", "state"])["candidatevotes"].rank(ascending=False)
#print(data.head(15))

#vytvoř novou tabulku, která bude obsahovat pouze vítěze voleb
#vitezove = data.sort_values(["rank"])
vitezove = data[data["rank"] == 1]
#print(vitezove)

#shift() - přidej nový sloupec, abys v jednotlivých řádcích měl(a) po sobě vítězné strany ve dvou po sobě jdoucích letech
vitezove_sorted = vitezove.sort_values(["state", "year"])
vitezove_sorted["previousWinnerParty"] = vitezove_sorted.groupby(["state"])["party_simplified"].shift(periods = +1)

#porovnej, jestli se ve dvou po sobě jdoucích letech změnila vítězná strana. Můžeš k tomu použít např. funkci numpy.where() nebo metodu apply()
vitezove_sorted = vitezove_sorted[vitezove_sorted['year'].map(lambda x: str(x)!="1976")]
vitezove_sorted["change"] = numpy.where(vitezove_sorted["previousWinnerParty"] == vitezove_sorted["party_simplified"], 0, 1)
#print(vitezove_sorted.head(20))

#proveď agregaci podle názvu státu a seřaď státy podle počtu změn vítězných stran
vitezove_sorted["change"] = vitezove_sorted["change"].astype('int64')
#print(vitezove_sorted.dtypes)
vitezove_sorted = vitezove_sorted.groupby(["state"])["change"].sum()
vitezove_sorted = pd.DataFrame(vitezove_sorted)
vitezove_sorted = vitezove_sorted.sort_values("change", ascending=False)
#print(vitezove_sorted)

#vytvoř sloupcový graf s 10 státy, kde došlo k nejčastější změně vítězné strany. Jako výšku sloupce nastav počet změn
# Sloupcový graf 10 států, kde došlo k nejčastější změně vítězné strany
TOP15StatesChanges = vitezove_sorted.head(15)
print(TOP15StatesChanges)

#TOP15StatesChanges = TOP15StatesChanges.set_index("state")
TOP10StatesChanges = TOP15StatesChanges.iloc[:10]
TOP10StatesChanges.plot(kind="bar")
#plt.show()
#plt.ylabel("Nr. of changes")
#plt.xlabel("States")
# Popisky pro legendu grafu
#plt.legend(['X', "Y"])



#2.cast ukolu - pracuj s tabulkou se dvěma nejúspěšnějšími kandidáty pro každý rok a stát (tj. s tabulkou, která oproti té minulé neobsahuje jen vítěze, ale i druhého v pořadí)
rank1a2 = data[data["rank"] <= 2]
#print(rank1a2)

#přidej do tabulky sloupec, který obsahuje absolutní rozdíl mezi vítězem a druhým v pořadí
rank1a2["secondCandidateVote"] = rank1a2.groupby(["year", "state"])["candidatevotes"].shift(periods = -1)
#print(rank1a2)

#přidej sloupec s relativním marginem, tj. rozdílem vyděleným počtem hlasů
rank1 = rank1a2[rank1a2["rank"] == 1]
rank1["margin"] = (rank1["candidatevotes"] - rank1["secondCandidateVote"]).div(rank1["candidatevotes"], level=None, fill_value=None)
#print(rank1a2)

#seřaď tabulku podle velikosti relativního marginu a zjisti, kdy a ve kterém státě byl výsledek voleb nejtěsnější
rank_sorted = rank1.sort_values(["margin"])#2000 FLORIDA REPUBLICAN 0,000184
#print(rank_sorted)

#vytvoř pivot tabulku, která zobrazí pro jednotlivé volební roky, kolik států přešlo od Republikánské strany k Demokratické straně,kolik států volilo kandidáta stejné strany.
kontingencni_tabulka = pd.pivot_table(data=rank_sorted, values="change", index="state", columns="party_simplified", aggfunc=sum, fill_value=0)
print(kontingencni_tabulka)
