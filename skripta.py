# Importovanje biblioteka
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
# --------------------------------- 1. --------------------------------------------------------------

# Definisati fazi promenljivu Temperatura, nad univerzalnim skupom opsega od 0 do 25°𝐶 i to 
# pomoću tri skupa: hladno, normalno i toplo, na osnovu sledećeg crteža

# Univerzalni skup
temperatura = ctrl.Antecedent(np.arange(0, 26, 1), 'temperatura')


# Definisanje funkcija pripadnosti
temperatura['hladno'] = fuzz.zmf(temperatura.universe, 5, 10)  
temperatura['normalno'] = fuzz.gaussmf(temperatura.universe, 12.5, 2.12) 
temperatura['toplo'] = fuzz.gaussmf(temperatura.universe, 25, 4)    

fig = temperatura.view()
ax = plt.gca()
ax.legend(['temperatura_hladno', 'temperatura_normalno', 'temperatura_toplo'], loc="center left")
plt.show()


# --------------------------------- 2. --------------------------------------------------------------
cena_goriva = ctrl.Antecedent(np.arange(2.5, 4.05, 0.05), 'cena_goriva')
cena_goriva['jeftino'] = fuzz.trapmf(cena_goriva.universe,[0,0,2.7,3.0])
cena_goriva['srednje'] = fuzz.trimf(cena_goriva.universe, [2.8, 3.2, 3.8])
cena_goriva['skupo'] = fuzz.trapmf(cena_goriva.universe, [3.4, 3.6, 4.0, 4.0])

fig = cena_goriva.view()
ax = plt.gca()
ax.legend(['gorivo_jeftino', 'gorivo_srednje', 'gorivo_skupo'], loc="center right")
plt.show()

# --------------------------------- 3. --------------------------------------------------------------
zarada = ctrl.Consequent(np.arange(0, 1400001, 10000), 'zarada')
zarada['mala'] = fuzz.trapmf(zarada.universe, [0, 0, 200000, 800000])
zarada['srednja'] = fuzz.trimf(zarada.universe, [400000, 900000, 1100000])
zarada['velika'] = fuzz.trapmf(zarada.universe, [1000000, 1200000, 1400000, 1400000])

fig = zarada.view()
ax = plt.gca()
ax.legend(['zarada_mala', 'zarada_srednja', 'zarada_velika'], loc="center left")
plt.show()

# Komentar: Ovde vidim da je dato po 3 terma za svaku lingvisticku promenljivu,
# sto znaci da je izabrana vrednost hiperparametra P iz formule "2P+1", jednaka 1.

# --------------------------------- 4. --------------------------------------------------------------
# Kreirati fuzzy sistem i dodati promenljive u sistem


# --------------------------------- 5. --------------------------------------------------------------
# a) ucitati fajl walmart.csv
data = pd.read_csv("Walmart.csv")

# b) generisati pravila koriscenjem wang mendel metode
# tako da na osnovu cene goriva i temperature odredjuje
# zaradu pumpe

rules = {}

# Prema Wang Mendel metodi, za svaki podatak iz skupa podataka walmart.csv,
# izracunacemo stepen pripadnosti za svaku fuzzy vrednost.
# Biramo najveci stepen pripadnost za svaku vrednost, znaci kome najvise pripada -> i to postaje label
# i na osnovu toga formiramo pravila : IF cenaGoriva je .. AND temperatura je .. THEN zarada je ..
for _, row in data.iterrows():
    temp_val = row['Temperature']
    fuel_val = row['Fuel_Price']
    sales_val = row['Weekly_Sales']

    # print(temp_val, fuel_val, sales_val)

    # Nalazimo dominantne fuzzy skupove (tamo gde ima najveci st. pripadnosti)
    # print(temperatura.terms)

    # -------- Temperatura ---------
    # [("hladno", 0.2), ("normalno", 0.8), ("toplo", 0.1)]
    temp_label, temp_mu = max(
        [(term, fuzz.interp_membership(
                    temperatura.universe, 
                    temperatura[term].mf, 
                    temp_val))
         for term in temperatura.terms],
        key= lambda x: x[1] # bira maksimum po drugoj vrednosti odnosno stepenu pripadnosti
        )
    
    # -------- Cena goriva ---------
    fuel_label, fuel_mu = max(
        [(term, fuzz.interp_membership(
                    cena_goriva.universe, 
                    cena_goriva[term].mf, 
                    fuel_val))
         for term in cena_goriva.terms],
        key=lambda x: x[1]
        )
    
    # ---------- Zarada -----------
    sales_label, sales_mu = max(
        [(term, fuzz.interp_membership(
                    zarada.universe, 
                    zarada[term].mf, 
                    sales_val))
         for term in zarada.terms],
        key=lambda x: x[1]
        )

    # Antecedent ce biti ulazi (fuel, temp), a consequent je sales
    antecedent = (fuel_label, temp_label)
    consequent = sales_label

    # Moramo da racunamo i stepen tacnosti (confidence factor) za svako pravilo
    # da bismo izbegli konflikte, tj. uzimacemo ono pravilo R za koje imamo najveci
    # stepen tacnosti, koji se racuna kao proizvod stepena pripadnosti antecedenata i konsekventa
    cf = fuel_mu * temp_mu * sales_mu

    if antecedent not in rules or rules[antecedent][1] < cf:
        rules[antecedent] = (consequent, cf)
    
    # Na ovaj nacin smo se resili i duplikata, jer nece dodavati ako vec postoji to pravilo.

# ------------ Kreiranje fuzzy pravila i dodavanje u sistem ---------------------------------------
print("\n--- Fuzzy pravila (Wang-Mendel) ---")
rule_objects = []
for (fuel, temp), (sales, cf) in rules.items():
    print(f"IF gorivo je {fuel} AND temperatura je {temp} "
          f"THEN zarada je {sales} "
          f"(CF = {cf:.3f})")
    rule_objects.append(ctrl.Rule(cena_goriva[fuel] & temperatura[temp], zarada[sales]))

system_ctrl = ctrl.ControlSystem(rule_objects) 
system_sim = ctrl.ControlSystemSimulation(system_ctrl)


# --------------------------------- 7. --------------------------------------------------------------
# Odgovoriti na pitanje "Kolika je zarada pumpe?", i nacrtati zaradu ako je:
# a) temperatura = 14 && cena goriva = 3
# b) temperatura = 20 && cena goriva = 3.5
# c) temperatura = 5 && cena goriva = 3.2

print("\n--------------- Pitanja i odgovori ----------------")

pitanja = [(14, 3), (20, 3.5), (5, 3.2)]
broj_pitanja = 0
for temperatura_pitanje, gorivo_pitanje in pitanja:
    broj_pitanja = broj_pitanja+1
    system_sim.input['temperatura'] = temperatura_pitanje
    system_sim.input['cena_goriva'] = gorivo_pitanje
    system_sim.compute()
    print(f"Temperatura={temperatura_pitanje}, Fuel={gorivo_pitanje} => Zarada: {system_sim.output['zarada']:.2f}\n")
    zarada.view(sim=system_sim)
    plt.savefig(f"Pitanje_{broj_pitanja}.png")


# --------------------------------- Dodatak ---------------------------------------------------------
# stampanje matrice kao sa predavanja kada smo pominjali Wang-Mendel metodu:
# Redovi - term-i za temperature
# Kolone - term-i za cene goriva

def rules_to_matrix(rules, temp_terms, fuel_terms):
    matrix = pd.DataFrame(
        index=temp_terms,   
        columns=fuel_terms  
    )

    # Popunjavamo matricu na osnovu pravila
    for (fuel, temp), (sales, cf) in rules.items():
        matrix.loc[temp, fuel] = sales
    
    return matrix

matrix = rules_to_matrix(rules, temperatura.terms, cena_goriva.terms)
print(matrix)

input("Pritisni Enter za izlaz...")