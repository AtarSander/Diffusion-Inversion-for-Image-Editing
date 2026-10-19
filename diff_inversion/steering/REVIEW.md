# Przegląd kodu steeringu — 19.10.2026

Branch `feature/steering` wydzielono z commita `7f173f5`, wspólnego wtedy dla
`feature/higher-cfg` i `feature/super-resolution-inversion`. Ten ostatni nie miał
własnych dodatkowych commitów. Niezapisane zmiany treningu, FLUX i innych
eksperymentów pozostają poza zakresem commita steeringu.

## Co zostało uporządkowane

Poprzedni kod łączył obliczenia, kontrakty cache’u, audyty plików i bibliotek,
obsługę klastra, publikowanie artefaktów i kolejne wersje konkretnych eksperymentów.
Przykładowo `code_inventory()` hashowało listę plików źródłowych, a `fingerprint()`
hashowało JSON konfiguracji lub kontraktu. To infrastruktura odtwarzalności dawnych
runów, nie element estymatora ani operatora steeringu.

Nowa ścieżka mieści się w `diff_inversion/steering/`. Moduły mają pojedyncze
zadania, a wspólny CLI realizuje `extract`, `fit`, `generate`. Zmiana autora,
liczby źródeł, promptów, seeda, reprezentacji, siły, warstw lub czasu działania
odbywa się przez konfigurację/listę obrazów. Nie powstaje skrypt na każdy run.

W nowej ścieżce nie ma `code_inventory`, `fingerprint`, list bibliotek, przypiętych
jobów ani audytów HPC. Pozostają zwykłe kontrole poprawności kształtu, wartości,
harmonogramu i zgodności parametrów modelu. Manifest podaje czytelne ustawienia,
prompty i wyniki. Hash dotyczy wyłącznie faktycznego szumu początkowego, zgodnie
z preferencją wspólnego startu wszystkich wariantów.

Stare `diff_inversion/eval/wikiart_steering.py`, runnery i zamrożone pakiety
pozostają lokalnie poza nowym commitem. Nie zmieniono ich obliczeń ani plików
użytych przez ukończone/zakolejkowane zadania. Nowy format `features.pt` oraz
`directions.pt` jest prostszy; paczki z odczytem swobodnej rekonstrukcji wymagają
ponownej ekstrakcji aktywacji i wyznaczenia wektorów.

## Co reprezentuje metoda

**Zmiana zbierania aktywacji, 19.10.2026:** jedyna ścieżka odczytu startuje
z kolejnych zapisanych stanów inwersji. Dla N kroków zapisujemy N+1 latentów,
a odczyt k podaje bazowemu UNetowi `states[N-k]` z właściwym timestepem.
Nie propagujemy wyniku poprzedniego odczytu do następnego. Usunięto opcję
`reader` i wariant swobodnej rekonstrukcji z nowego kodu. To zmiana sposobu
pozyskiwania aktywacji względem pierwszej konfiguracji nowego interfejsu;
wektory z tamtej ścieżki wymagają ponownej ekstrakcji i wyznaczenia.
Historyczne wyniki opisane poniżej zachowują swój rzeczywisty protokół.

Najważniejszą różnicą względem promptowego CASteer jest pochodzenie aktywacji:
u nas pochodzą z realnych obrazów odwracanych DDIM lub MW, następnie czytanych
bazowym UNetem. Nazwa metody opisuje źródło wektora; docelowa generacja w obu
ramionach korzysta z tego samego modelu, promptu i rzeczywistego szumu.

Kierunki różnicy średnich są osobne dla reprezentacji, metody inwersji, warstwy
i kroku odszumiania. Odczyt obejmuje wszystkie znalezione bloki i kroki; osobno
wybieramy, gdzie i kiedy wstrzykiwać offset. `block` to przyrost pojedynczego
`BasicTransformerBlock`, czyli SA + CA + FF wraz z ich otoczeniem i residualami,
a nie cały blok upsamplingu UNetu.

SA i block częściowo obejmują te same obliczenia. Nie są równoważne: zmiana SA
przechodzi jeszcze przez CA i FF, a zmiana wyjścia bloku jest dodawana po nich.
Zbliżone obrazy nie dowodzą redundantności operatorów. Wspólna skala RMS
wyrównuje żądaną dawkę w aktywacjach, nie wielkość zmiany obrazu ani stylu.

`last36` ma konkretny precedens: CASteer v4, Appendix H.5, Table 26, erasure stylu
Van Gogha w CA SDXL. Przeniesienie tego zakresu na SA/block i dodawanie stylu
z realnych obrazów jest naszą adaptacją. Nie stanowi dowodu optymalności tej
konfiguracji. Dokładne źródło i wyniki opisuje weekly.

## Co wynika z dotychczasowych wyników

W pilocie VG użyto 8+8 obrazów konstrukcyjnych i osobnych 4+4 walidacyjnych.
N8 było ustawioną wielkością pilota; wcześniej były też próby 100+100 i 200+200.
Wyboru N8 nie oparto na przewadze MW nad DDIM. Nie należy porównywać tych serii
jako kontrolowanej zmiany samego N: zmienił się również dobór źródeł i protokół.

SA dawało najbardziej obiecujący częściowy sygnał specyficzności. Duże widoczne
zmiany block przy lambda 2 nie wystarczają: kontrola swap bywała równie silna
lub silniejsza. Ranking największych zmian nie jest rankingiem najlepszego stylu.

MW poprawiał średni PSNR rekonstrukcji w kohorcie źródłowej o około 0,38 dB.
Niezależna walidacja steeringu nie spełniła przyjętej reguły przewagi MW po
uwzględnieniu własnej kontroli swap. Sąsiednie kroki mają podobne kierunki,
ale w dłuższym przedziale występuje dryf; dotychczasowa analiza czasowa dotyczyła
osobnej kohorty Mashkov–Cézanne, nie VG. Brakuje przyczynowego testu stosowania
zamrożonego wektora w innych krokach.

## Sprawdzenie reorganizacji

Po zmianie na odczyt wyłącznie ze stanów inwersji przechodzi **20 testów CPU**.
Sprawdzają m.in. dokładne przypisanie wszystkich 50 odczytów do stanów i timestepów
DDIM/MW, wyłączenie LoRA podczas odczytu oraz brak kroku schedulera w tej fazie.
Zmiana jednego zapisanego stanu wpływa wyłącznie na odpowiadający mu odczyt.
Test integracyjny sprawdza zapis trajektorii, a osobny test odrzucanie paczek
aktywacji i kierunków sprzed tej zmiany. Ruff również przechodzi.

Pierwsza wersja interfejsu przeszła **17 testów CPU**, również w izolowanej kopii zawierającej
wyłącznie pliki bazowego commita i wybrany zakres steeringu, bez lokalnego
archiwum. Obejmują rzeczywiste małe `BasicTransformerBlock` i schedulery
Diffusers: granice hooków, FP32 różnicę output–input, momenty przed poolingiem,
zmianę wyłącznie gałęzi conditional, zerową dawkę, indeksowanie stanów,
DDIM/MW i przywracanie LoRA, wspólny RMS, kontrole oraz pełny przepływ trzech
etapów. Osobny test sprawdza flagi rzeczywistych warstw PEFT. W teście integracji
każdy seed powtarza całą macierz promptów i wariantów na wspólnym szumie.

Przy pierwszej reorganizacji lokalne porównanie ze starymi prymitywami dało **196 zgodnych porównań,
maksymalny błąd 0**: odczyty CA/SA/block, ich drugie momenty i wynik epsilon
na małym transformerze oraz skalowanie dawek na syntetycznych kierunkach.
To ograniczone sprawdzenie arytmetyki; nie jest pełną zgodnością obrazów SDXL.
Nie odtworzono pretrained SDXL, historycznych generatorów RNG ani wszystkich
starych ablacji. Nie uruchamiano nowych eksperymentów GPU ani zadań klastra.

Nowy interfejs nie obejmuje automatycznej selekcji źródeł, metryk CSD/DINO,
treningu, forward-diffusion FD2, projekcyjnego erasure ani pełnej historycznej
macierzy ablacji. Te elementy nie są ukrytymi zależnościami prostej ścieżki.

Instrukcje i format danych: [README](README.md). Szczegóły pilota, źródeł,
promptów i historycznych obrazów: [weekly VG](../../docs/weekly/2026-10-12_2026-10-18_van_gogh_steering.md).
Pliki `reports/` i modele pozostają lokalne; brak ich w samym checkoutcie Git
nie oznacza nowych wyników ani utraty oryginalnych artefaktów.
