import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

data_countries = pd.read_csv("countries.csv")
data_ukol2a = pd.read_csv("ukol_02_a.csv")
data_ukol2b = pd.read_csv("ukol_02_b.csv")

##1.Inflace - první části úkolu můžeš pracovat se všemi státy nebo jen pro státy EU - záleží na tobě
#print(data_countries)
#print(data_ukol2a)

##Test normality dat: 
    #Nulová hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy v našem souboru mají normální rozdělení.
    #Alternativní hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy v našem souboru nemají normální rozdělení.

res97 = stats.shapiro(data_ukol2a["97"])
res98 = stats.shapiro(data_ukol2a["98"])
print(res97)
print(res98)
##Obě pvalue jsou větší než 0.05 tzn. data mají normální rozdělení.

##Pro test hypotézy můžeme použít párový t-test. Test předpokládá, že data mají normální rozdělení.
##Hypotézy:
##H0: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy se nezměnilo mezi létem 2022 (sloupec 97) a zimou 2022/2023 (sloupec 98).
##H1: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy se změnilo mezi létem 2022 (sloupec 97) a zimou 2022/2023 (sloupec 98).
res = stats.ttest_rel(data_ukol2a["97"], data_ukol2a["98"])
print(res)
##p-value není větší než 0.05 tzn. že zamítáme nulovou hypotézu a procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy se změnilo 
##mezi létem 2022 (sloupec 97) a zimou 2022/2023 (sloupec 98).

##2.Důvěra ve stát a v EU - ve druhé a třetí části pracuj pouze se státy EU.
#print(data_ukol2b)
#Inner join - získám pouze státy EU
data_countriesEU = pd.merge(data_countries, data_ukol2b, on="Country", how='inner')
print(data_countriesEU)

##Test normality dat: 
    #Nulová hypotéza: Procento lidí, kteří dané instituci věří (vláda/EU) v našem souboru mají normální rozdělení.
    #Alternativní hypotéza: Procento lidí, kteří dané instituci věří (vláda/EU) v našem souboru nemají normální rozdělení.

resGovernment = stats.shapiro(data_countriesEU["National Government Trust"])
resEU = stats.shapiro(data_countriesEU["EU Trust"])
print(resGovernment)
print(resEU)
##Obě p-value jsou větší než 0.05 tzn. data mají normální rozdělení.

##Test založený na Pearsonově korelačním koeficientu: Test řeší, zda je zjištěná korelace statisticky významná. 
##H0: Procento lidí, kteří věří vládě a procento lidí, kteří věří EU nejsou statisticky závislé.
##H1: Procento lidí, kteří věří vládě a procento lidí, kteří věří EU jsou statisticky závislé
res = stats.pearsonr(data_countriesEU["National Government Trust"], data_countriesEU["EU Trust"])
print(res)
##p-value není větší než 0.05 tzn. že zamítáme nulovou hypotézu a procento lidí, kteří věří vládě a procento lidí, kteří věří EU jsou statisticky 
##závislé.

##3.Důvěra v EU a euro
data_countries_eurozone = data_countriesEU[data_countriesEU["Euro"] == 1]
data_countries_OUTeurozone = data_countriesEU[data_countriesEU["Euro"] == 0]
print(data_countries_eurozone)
print(data_countries_OUTeurozone)

##Pro test hypotézy můžeme použít nepárový t-test. Test předpokládám, že data mají normální rozdělení.
#H0: Procenta lidí, kteří věří EU a žíjí ve státech platících Eurem je stejné jako procenta lidí, kteří věří EU a nežíjí ve státech platících Eurem.
#H1: Procenta lidí, kteří věří EU a žíjí ve státech platících Eurem není stejné jako procenta lidí, kteří věří EU a nežíjí ve státech platících Eurem.
res = stats.ttest_ind(data_countries_eurozone["EU Trust"], data_countries_OUTeurozone["EU Trust"])
print(res)
##p-value je větší než 0.05 tzn. že potvrzujeme nulovou hypotézu a procento lidí, kteří věří v EU je stejné jako procenta lidí, kteří věří EU a nežíjí 
##ve státech platících Eurem.