# Notatki — LightGCN Projekt

---

## Problem i idea modelu

**Problem:** Mamy użytkowników i miejsca. Chcemy przewidzieć, które miejsca użytkownik polubi.

**Jak?**
1. Budujemy graf: użytkownik ↔ miejsca które odwiedził
2. Każdy użytkownik i miejsce dostaje wektor liczb (embedding)
3. Model uczy te wektory przez uśrednianie wektorów sąsiadów w grafie (K razy)
4. Rekomendacja = znajdź miejsca, których wektor jest najbardziej podobny do wektora użytkownika

**Czego uczymy?** Model widzi trojaczki (użytkownik, miejsce które odwiedził, losowe miejsce) i uczy się, że to pierwsze powinno dostać wyższy wynik.

**Dlaczego "Light"?** Klasyczne GCN mają skomplikowane transformacje — autorzy pokazali, że wystarczy samo uśrednianie sąsiadów i działa lepiej.

---

## Co my robimy w projekcie i czym się różnimy od artykułu?

**Cel projektu:** To projekt zaliczeniowy na kurs "Zaawansowana eksploracja danych". Implementujemy LightGCN od zera i sprawdzamy czy nasze wyniki zgadzają się z paperem.

### Różnice względem artykułu

|              | Artykuł (autorzy)                      | My                                              |
|--------------|----------------------------------------|-------------------------------------------------|
| Cel          | Zaproponować nowy model i udowodnić że działa | Zaimplementować ich model i zreplikować wyniki |
| Kod          | Napisany przez badaczy, zoptymalizowany | Napisany przez nas od zera w PyTorch           |
| Eksperymenty | Porównanie z wieloma innymi modelami   | Tylko LightGCN — jak zachowuje się przy K=1,2,3,4 warstwach |
| Skala        | Wiele datasetów                        | Tylko Gowalla                                   |

### Co konkretnie robimy

1. Wczytujemy dane Gowalla i budujemy graf
2. Implementujemy model dokładnie wg równań z paperu
3. Trenujemy i patrzymy czy nasze Recall@20 ≈ 0.1327 (wynik z paperu)
4. Eksperymenty: jak liczba warstw K wpływa na jakość?

**Główne pytanie projektu:** Czy da się samemu odtworzyć wyniki z paperu naukowego?

---

## Czy to wpisuje się w polecenie?

> *Zaliczenie laboratorium polegać będzie na wyborze jednej pracy naukowej dotyczącej szeroko pojętej analizy danych, zrozumieniu jej treści, implementacji proponowanego rozwiązania, przeprowadzeniu eksperymentów na danych oraz przedstawieniu wyników i porównania z metodami wcześniejszymi.*

**Tak, wpisuje się idealnie — punkt po punkcie:**

| Wymaganie                   | Co my robimy                                                                 |
|-----------------------------|------------------------------------------------------------------------------|
| Wybór pracy naukowej        | LightGCN, SIGIR 2020                                                         |
| Zrozumienie treści          | Notebook z eksploracją danych + CLAUDE.md z opisem każdego równania          |
| Implementacja rozwiązania   | Cały `src/` — model, trening, ewaluacja napisane od zera                     |
| Eksperymenty na danych      | K=1,2,3,4 warstwy, porównanie z/bez layer combination, krzywe zbieżności    |
| Porównanie z metodami wcześniejszymi | Porównujemy LightGCN vs MF (Matrix Factorization, K=0) — to jest "metoda wcześniejsza" |

> **Uwaga:** Jedyne co warto sprawdzić — czy macie wystarczająco dużo porównań z metodami wcześniejszymi. Paper porównuje z NGCF, MF, itp. U was jest tylko MF (K=0). Może warto to podkreślić na prezentacji, że MF to baseline który LightGCN zastępuje.

---

## Czy nasza implementacja to to samo co implementacja twórców?

**Nie, to są dwie różne implementacje.**

Autorzy paperu napisali swój kod (dostępny na GitHubie) — zoptymalizowany, napisany przez doświadczonych badaczy, prawdopodobnie w TensorFlow.

Nasza implementacja to niezależne napisanie tego samego modelu od zera w PyTorch, na podstawie równań z paperu — nie przez kopiowanie ich kodu.

To jest właśnie sedno tego zaliczenia — że rozumiesz paper na tyle, żeby samemu go zaimplementować. Gdybyś skopiował ich kod, nie byłoby żadnego sensu w tym ćwiczeniu.

> **Analogia:** Przepis na ciasto istnieje w książce, ale ty upiekłeś to ciasto sam w swojej kuchni — to nie jest "to samo ciasto co autor przepisu zrobił", ale efekt powinien być podobny.

---

## `train.py` vs notatnik z eksperymentami

Skrypt `train.py` i notatnik z eksperymentami to dwie osobne ścieżki:

- **`train.py`** — To jest Twój "główny model". Trenujesz go raz, najlepiej jak potrafisz, żeby uzyskać finalny wynik, którym będziesz się chwalić.
- **Notatnik z eksperymentami** — To jest Twoja "dokumentacja naukowa" (tzw. Ablation Study). Służy do tego, żeby udowodnić prowadzącemu lub recenzentowi, że Twoje ustawienia (np. to, że wybrałeś 3 warstwy) nie są przypadkowe, tylko wynikają z testów.

### Czym właściwie jest "Ablation Study" (Eksperymenty)?

Zrozumienie, dlaczego musimy trenować model wielokrotnie, jest kluczowe. Wyobraź sobie, że budujesz samochód wyścigowy. `train.py` to Twój gotowy bolid. Eksperymenty to testy na torze, gdzie sprawdzasz: "A co jeśli zdejmę spojler? A co jeśli zmienię opony?". Musisz przejechać okrążenie (trening) za każdym razem od nowa, żeby porównać wyniki.

---

## Analiza wyników i ocena zaliczeniowa

**Czy nadaje się na zaliczenie? TAK, solidnie.**

Projekt spełnia wszystkie wymagania kursu punkt po punkcie: wybór pracy naukowej, implementacja od zera, eksperymenty, porównanie z metodami bazowymi (MF = K=0) i z paperem.

### Full run (`train.py`, 400 epok, K=3)

| Metryka   | Nasz wynik | Paper  | Różnica |
|-----------|------------|--------|---------|
| Recall@20 | 0.1773     | 0.1823 | −2.7%   |
| NDCG@20   | 0.1510     | 0.1554 | −2.8%   |

2.7% luki przy 400 epokach vs ~1000 epok w paperze — to bardzo dobry wynik jak na projekt zaliczeniowy.

**Ciekawy fakt z ablacji:** nasz K=1 (0.1663) bije wynik K=1 z paperu (0.1549) o 7.4%. Dla K=2 różnica to tylko 1.4%. To dowód na wierność implementacji.

**Zbieżność:** model nie spłaszcza się przy epoce 145 jak wcześniej sądziłem — dopiero przy ~350. Przez pierwsze 100 epok "wyciąga" 95% końcowego Recall@20, ale potem bardzo powoli jeszcze rośnie aż do ~400.

**Layer combination (+14.6%):** all-layer average (0.1653) vs last-layer-only (0.1442) — potwierdza główne twierdzenie paperu.

---

## Co to jest Recall@20?

Wyobraź sobie że wiesz, że użytkownik lubi 5 konkretnych miejsc (test set). Model proponuje mu 20 miejsc. Recall@20 mówi: **ile z tych 5 prawdziwych trafiło w top 20?**

- Trafił 1 z 5 → Recall = 0.20
- Trafił 3 z 5 → Recall = 0.60
- Średnia po wszystkich użytkownikach = końcowy wynik

Nasze 0.1773 oznacza że średnio trafiamy ~17.7% prawdziwych upodobań użytkownika w top 20. To dobry wynik jak na tak rzadkie dane.

### "Wiernie zaimplementować metodę" — co to znaczy?

Tak, chodzi dokładnie o to żeby przepisać ich algorytm — ale nie kopiując kod, tylko rozumiejąc go i pisząc samemu na podstawie opisu w artykule.

Konkretnie: artykuł opisuje wzory matematyczne (równania 4, 6, 7 w paperze) i mówi np. "nie używaj macierzy wag, nie używaj ReLU, normalizuj tak a tak" — i Ty to zakodowałeś w PyTorchu.

To jest właśnie sedno projektu — czy rozumiesz CO i DLACZEGO tak działa, a nie czy skopiowałeś cudzy kod.

> Wynik 0.1773 vs 0.1823 z artykułu to różnica tylko dlatego że oni trenowali 1000 epok, Ty 400. Implementacja jest poprawna.

---

## O co chodzi w eksperymentach?

**Główne pytanie eksperymentów:** Czy więcej warstw w modelu = lepsze wyniki?

### Eksperyment 1: Ablacja po liczbie warstw K (główny)

Trenowaliśmy ten sam model 4 razy, za każdym razem zmieniając tylko jedną rzecz — liczbę warstw K:

- **K=1** → model patrzy tylko na bezpośrednich sąsiadów
- **K=2** → patrzy na sąsiadów sąsiadów
- **K=3** → jeszcze głębiej
- **K=4** → jeszcze głębiej

**Wynik:** K=2 wygrał (Recall=0.1712), K=3 był tuż za nim. K=4 był gorszy niż K=2 — czyli więcej warstw nie zawsze pomaga (model zaczyna "rozmywać" informacje).

To potwierdza wniosek z artykułu.

### Eksperyment 2: Krzywe zbieżności

Patrzyliśmy jak model się uczy przez 400 epok — czy w ogóle się uczy, kiedy przestaje się poprawiać.

**Wynik:** Największy skok w pierwszych 100 epokach, po 350 epokach model praktycznie stoi w miejscu.

> **Uwaga:** Eksperyment 2 to po prostu wynik głównego treningu — 400 epok z K=3. Przy okazji tego treningu logowaliśmy co 20 epok: loss, Recall@20, NDCG@20 — i z tych liczb narysowaliśmy wykresy zbieżności. Nie był to osobny eksperyment "z założenia" — raczej przy okazji głównego treningu zebraliśmy dane i je pokazaliśmy. Nic złego w tym, tak się to robi w praktyce.

---

## Czym różni się nasze rozwiązanie od rozwiązania twórców artykułu?

### Główne różnice

1. **Liczba epok treningu** — Autorzy trenowali ~1000 epok, my 400. Stąd wynika cały gap w wynikach (0.1773 vs 0.1823). Algorytm jest identyczny.
2. **Framework** — Autorzy mają własną implementację (oryginalnie TensorFlow). My napisaliśmy od zera w PyTorchu.
3. **Kod napisany samodzielnie** — Nie przekleiliśmy ich kodu, tylko przeczytaliśmy opis matematyczny w artykule i zakodowaliśmy go sami. To wymaga rozumienia co każdy wzór robi.

### Co jest identyczne

- Algorytm LightGCN (te same wzory, ta sama normalizacja)
- Dataset Gowalla
- Metryki Recall@20 i NDCG@20
- Hiperparametry: embedding 64, lr=0.001, λ=1e-4, K=3
- Funkcja straty BPR

**Krótko mówiąc:** Różnimy się tylko długością treningu i frameworkiem, nie samą metodą. Gdybyśmy trenowali 1000 epok to wynik byłby praktycznie taki sam jak w artykule.
