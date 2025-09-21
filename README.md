# Wang-Mendel Fuzzy Inference

Ovaj projekat implementira **Wang-Mendel metodu** za generisanje fuzzy pravila iz podataka i isprobava je nad skupom podataka `Walmart.csv` korišćenjem biblioteke **scikit-fuzzy**.

## 📌 Opis zadatka
Zadatak se nalazi u fajlu **Domaci 4.pdf**.  
Cilj je da se definišu fuzzy promenljive, generišu pravila metodom Wang-Mendel i kreira fuzzy sistem za zaključivanje.

## ⚙️ Instalacija i pokretanje

1. Klonirati repozitorijum:
   
   ```bash
   git clone https://github.com/marijajolovic/wang-mendel-fuzzy-inference.git
   cd wang-mendel-fuzzy-inference
   ```

2. Kreirati i aktivirati virtuelno okruženje:
   
    Na Windows-u: 

    
        python -m venv venv
        venv\Scripts\activate
        

    Na Linux-u:

        
        python3 -m venv venv
        source venv/bin/activate
        

3. Instalirati zavisnosti:
   
    ```
    pip install -r requirements.txt
    ```

4. Pokrenuti skriptu:
 
   ```bash
   python skripta.py
    ```

👤 Autor

Marija Jolović 46-2021

marija.jolovic@pmf.kg.ac.rs

Za potrebe predmeta **Inteligentni Sistemi**.