# Moonwalker vs DDIM — wektory stylu z WikiArt

## 2026-10-13 — primary: wielokrokowy steering aktywacji

Użytkownik doprecyzował, że docelowo chodzi o dodawanie λ razy wektor stylu
do wewnętrznych reprezentacji przez pierwsze kilka kroków odszumiania.
Wprowadzono osobny wariant `activation_window`; poprzedni `z_T` i jednorazowe
`h:mid:30` / `h:up:30` są zachowanym porównaniem historycznym.

| Rola | Faktyczny moduł SDXL | Okno |
|---|---|---|
| Główny wariant cross-attention | `up_blocks.0.attentions.0.transformer_blocks.0.attn2` | pierwsze5 |
| Główny wariant cross-attention | ten sam moduł | pierwsze10 |
| Kontrola bottleneck | `mid_block` | pierwsze10 |

Oba miejsca mają1280kanałów; cross-attention zbieramy po wyjściu całego
modułu attn2, bottleneck po wyjściu mid_block. To różne przestrzenie.
Wyjścia przestrzenne/tokenowe uśredniamy do kanałów, a podczas generacji
dodatek jest broadcastowany do pozycji tej samej warstwy. Cross-attention
jest kandydatem początkowym, nie udowodnioną najlepszą warstwą stylu.

Osobno dla każdego czasu i metody:

```text
d[m,layer,t] = mean(h_base(z_t_m(Van Gogh),t,"A painting"))
             - mean(h_base(z_t_m(matched other artists),t,"A painting"))
delta[m,layer,t] = lambda * s_ref[layer,t] * d[m,layer,t] / max(RMS(d),epsilon)
h_new[layer,t,p] = h_live[layer,t,p] + delta[m,layer,t]
```

`p` to pozycja przestrzenna/tokenowa. λ pozostaje stała we wszystkich
wywołaniach okna. Wektor i wspólna dla metod skala zależą od czasu; nie
normalizujemy względem każdego zdjęcia testowego. Rodzina pierwszych10
ma kształt10×1280 na warstwę/metodę. Okno5 korzysta z jej tego samego prefiksu.
Probing i cała generacja mają LoRAOFF; tylko inwersja Moonwalkera ma LoRAON.
Kierunek modyfikuje faktycznie bieżącą trajektorię, która po oknie jest dalej
odszumiana. Hooki są usuwane także po błędzie. λ0 dokładnie odtwarza bazową
generację. Ingerencja nie dodaje kolejnych wywołań denoisingu.

Pierwsze kroki liczymy **szum→obraz**. Wywołania0..4 to stany50..46 i czasy
981,961,941,921,901; wywołania0..9 to stany50..41 i czasy981..801. Są wyznaczane
z `state_map.denoising_calls`, nie ze zgadywanej arytmetyki indeksów.
[Plan okien, PNG](../../reports/technical/wikiart_steering_activation_window_20261013_v1/activation_window_plan.png)
i [PDF](../../reports/technical/wikiart_steering_activation_window_20261013_v1/activation_window_plan.pdf)
są wizualizacją protokołu, bez nowych wywołań modelu lub wyników jakości.

Nadal używamy tego samego native SDXL/checkpointu attention-only,100 dodatnich,
100dopasowanych i100losowych referencji konstrukcyjnych,12seedów oraz12zdjęć
walidacyjnych i12niezależnych zdjęć testowych. λgrid to[-.1,0,.02,.05,.1,.2].
Porównanie zdjęć zachowuje4ramiona referencje×inwersjaDDIM/Moonwalker. Obowiązują
kontroleλ0/losowa/permutowana/przeciwna i oddzielny baseline tekstowy.

Pierwotny N100 zakończył generację i scoring1764wierszy na24źródłach,
ale reguła selekcji zwróciła`no_selection`03:38CEST. W obu Moonwalker h30
brakowało dodatniego średniego przyrostuCSD, więc globalny wspólny zakres
trzech kluczy wynosił[0,0]. Zależny test nie wystartował. Jest to wynik bramki
historycznej jednokrokowej konfiguracji, nie wynik nowego okna. Archiwum:
[raport N100](../../reports/results/wikiart_steering/n100_single_step_20261013_v1/README.md),
[rzeczywiste obrazy/metyki](../../reports/results/wikiart_steering/n100_single_step_20261013_v1/html/pilot_validation.html),
[audyt](../../reports/results/wikiart_steering/n100_single_step_20261013_v1/audit.json).

Nowa reguła jest zamrożona przed nowymi wynikami: wybór tylko międzyup5/up10,
a mid10 jest z góry kontrolą diagnostyczną. Wspólny dodatni zakres CSD oraz
średnia DINO są porównywane symetrycznie między tymi dwoma oknami i oboma
kierunkami referencji. Zachowujemy pełną wspólną λgrid, bez ekstrapolacji.
Nie zmieniono interpretacji historycznego braku wyboru. Pole`selection_keys`
jest jawnie zapisane w konfiguracji/kontrakcie i nie pozwala ręcznie promowaćmid10.

## Uzasadnienie z publikacji i kolejność rozszerzeń

- [CASteer, §3 oraz AppendixI.2/I.3](https://arxiv.org/html/2503.09630v4):
  najbliższy mechanizm — różnice średnich cross-attention przypisane warstwie
  i czasowi, dodawane podczas odszumiania. Zaczynamy od jednego modułu i dwóch
  okien. Oryginał obejmuje wiele warstw oraz inne źródło przykładów/normalizację;
  nasze real-image inversion i relative-RMS są jawną adaptacją. Ich intensywność
  transferu określa głębokość inwersji, nie nasze K ani λ.
- [Haas et al., §V i AppendixF](https://arxiv.org/html/2303.11073v2):
  motywuje różnicę średnich w wewnętrznym h-space oraz pozyskiwanie kierunków
  z rzeczywistych obrazów przez inwersję. To uzasadnienie kontroli bottleneck.
  Wyniki dotyczą innych modeli/zadań; nie dowodzą jakości naszego SDXL stylu.
- [Through Van Gogh’s Eyes, §3.3–3.4](https://lakeeye1220.github.io/Global_Style_Transfer/static/paper.pdf):
  uczy funkcję MLP przewidującą offset bottleneck i dodaje guidance zachowania
  treści. Warto rozważyć później; wymaga osobnego treningu i dodatkowej ingerencji,
  przez co zmieniłby obecny eksperyment porównujący same metody inwersji.
- [SAeUron, §5.1, AppendixC/H](https://arxiv.org/html/2501.18052):
  przyszły wariant interpretuje aktywacje przez sparse autoencoder i ingeruje
  w wybrane cechy. AppendixH pokazuje dodatni steering cech stylu. Wydane wagi
  dotyczą innej bazy SD1.5 i hooka`unet.up_blocks.1.attentions.2`; samo1280wejść
  nie certyfikuje zgodności z naszym SDXL/attn2 ani z pooled kanałami. To tylko
  propozycja, zgodnie z poleceniem użytkownika pozostawiona na później.

Możliwe przyszłe testy SAE: rekonstrukcja pełnych tokenów z kontrolą zachowania
residuum, wybór cech VG względem dopasowanych negatywów tylko na construction/val,
sparowane wzmacnianie i wygaszanie tych samych decoderfeatures oraz losowe cechy
o dopasowanej RMS. Osobno można porównać naszą różnicę średnich z projekcją na
podprzestrzeń wybranych cech. To hipotezy badawcze; nie uruchomiono nowego SAE,
MLP, guidance ani treningu. Obecny cache kanałowych średnich nie zawiera pełnych
tokenów potrzebnych SAE.

## Wdrożenie okien, 2026-10-13 08:27 CEST

Implementacja przeszła193 testy w41,98s oraz Ruff. Pokryto rzeczywisty pipeline
CPU build→window seed/photo edit→evaluation, λ0, strict brak/mismatch kierunku
per-time, conditional branch, LoRAOFF, ciągłość trajektorii, cleanup i wybór/test
z kompletnym mappingiem okna. Receipt:
`reports/technical/wikiart_steering_auto_20261012_v1/window_20261013_tests.xml`.
Nowe profile: `wikiart_steering_window_{smoke,pilot}.yaml`;
koordynator wymaga jawnego`--protocol activation_window`, historyczny default
pozostaje jednokrokowy. Stan41..50 wyliczamy od nowa, bo stary cache nie ma49..41.

Immutable snapshot`code_20261013_windows_v1`:229plików, inventorySHA256
`6d33962ff2fc8b85c995c96cf96f85ff944d9a2504a3e2b84444a2fe192a01a8`.
Nowy numeryczny dowód oficjalnych evaluatorów przeszedł pod nowym hash metrics;
stary proof i wagi pozostały zachowane. Workflow:
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/autoflow_20261013_windows_v1`,
PID1556703, na08:27 trwaprepare_pilot_once bezbłędu. Ponownie użyto tylko
manifestów, po sprawdzeniu identyczności modułu danych i promptów z v8.
Limity2/12/4h i N100 nie rosną; work-loop budget119400pilot +3runtimeprobe
U-Neta. Nie dodajemyNFE podczas denoisingu, adapter/module search lub N200.
ObserverPID1559901 czeka na przyszłegeometry/validation/test HTML pod
`wikiart-steering/reports/interactive_windows_20261013_v1` i preferujeoknowe
aktywacje w podglądzie. Nie ma jeszcze wyniku jakości nowego wariantu.

Aktualizacja08:30 CEST: CPUbramki przeszły; rzeczywiście zgłoszono o08:27:34–35
**smoke3362255 → pilot3362256 → test3362257** z zależnościami afterok.
Koordynator ma `running/gpu_dag`, bez błędu. Lokalny plan/launch/numericalproof
w `reports/technical/wikiart_steering_activation_window_20261013_v1/`.
Nie utożsamiaj tych ID ani ich przyszłych wyników z dawnym jednokrokowym N100.
08:32 CEST: smoke3362255 rzeczywiście działa na t0007, etapedit_seed_validation.
GPUekstrakcja/budowa i diagnostyka wektorów zakończone; kontrakt v3 zawiera
każdy czas981..801 i odpowiadający mu kierunek. Pilot/test pozostajądependent.
To potwierdzenie wykonania technicznego, bez wyniku jakości okien.
[Rzeczywista geometria nowych wektorów N2 i dokładne okna](../../reports/technical/wikiart_steering_activation_window_20261013_v1/activation_window_smoke_geometry_20261013_v1.html)
powstała z ukończonej publikacji GPU; sprawdzono272artefakty diagnostyczne.
Zawiera czasy początkowe/końcowe h:up:50/46/41 oraz h:mid:50/41,
z tabelą całej rodziny wektorów. To geometria technicznaN2, bez oceny jakości.

**Najnowszy potwierdzony stan, 2026-10-12 22:58 CEST:** pełne dane i audyt
kohort są gotowe, a rzeczywisty smoke SDXL zakończył się `0:0`. Job **3361924**
działał 22:19:08–22:46:00 (26:52) i opublikował 246 wierszy ewaluacji oraz
206 unikalnych ścieżek pikseli obrazów/źródeł/rekonstrukcji. Pilot N100
**3361925** rozpoczął się 22:57:33 na t0021 i wykonuje ekstrakcję referencji;
test **3361926** ma zależność po pilocie.
N2 jest testem technicznym; jego scoring CSD/DINO celowo jest unavailable.
Oficjalne evaluatory i niezależny sanity autora są gotowe dla pilota.
Poniższe wpisy o wcześniejszych stanach zachowano jako historię.

[Interaktywny raport po polsku, offline](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/explainer_actual_transfer_v3.html)
zawiera rzeczywiste obrazy obu zdjęć, przełączanie zapisanych α/warstw/kontroli,
geometrię dwóch kierunków, wykresy i galerię użytych referencji. Suwak geometrii
oblicza amplitudę już zapisanych kierunków; nie generuje nowych obrazów ani
nie interpoluje pikseli. Wybrano dwa pierwsze alfabetyczne ID, bez rankingu jakości.
[Podgląd pierwszego zdjęcia z_T α=0,05](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/photo_preview_v3/photo_preview.png),
[galeria wybranych prac pilota](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/reference_gallery/pilot/gallery.pdf),
[pełne wyniki per-obraz](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/evaluation_v3/validation/per_image.csv).
HTML sprawdził 246 wierszy, 206 unikalnych ścieżek pikseli i zgodność kontraktu
z geometrią; obrazy są miniaturami istniejących wyników, nie ilustracjami AI.

## Zakończony audyt danych i znaczenie etykiety autora

Transfer wszystkich 72 przypiętych shardów Parquet plus metadanych trwał
18:50:06–19:05:13 CEST, około 15 min 7 s: 33 735 356 963 bajty (33,735 GB).
Zgodne są wszystkie upstream LFS SHA256 i rozmiary. Eksport zachował 81 444
oryginalne pliki obrazowe; późniejszy prepare zdekodował i wyliczył SHA/pHash
dla każdego z nich. Pobranie zakończyło się poprawnie, ale źródłowe etykiety
wymagały filtracji. 79 434 heurystyczne grupy dzieł nie są certyfikowanymi
identyfikatorami wszystkich możliwych reprodukcji i cropów.

| Etap / kategoria wykluczenia | Liczba |
|---|---:|
| Surowe rekordy | 81 444 |
| Kwalifikujące się po filtracji i deduplikacji | 36 419 |
| Van Gogh kwalifikujący się / surowe etykiety | 1 724 / 1 889 |
| Znani inni autorzy kwalifikujący się | 34 695 |
| Nieznany autor (rozłączne przypisanie przyczyny) | 41 892 |
| Nieznany genre | 2 663 |
| Deduplikacja | 292 |
| Niejednoznaczny genre | 60 |
| Sprzeczni autorzy w grupach pHash | 116 |
| Sprzeczni autorzy identycznych bajtów | 2 |

Łącznie wykluczono 45 025 rekordów; powyższe przyczyny są rozłączne według
kolejności filtracji, a surowe braki etykiet nakładają się. Kwarantanna obejmuje
39 całych grup / 118 członków, bez prac van Gogha; nie rozstrzygano autora
głosowaniem ani CSD. Główne negatywy dopasowano dokładnie rozkładem genre,
limit autora 20/100 (referencje 4/20). Zapisany manifest N100 ma SHA256
`f9df9bdc81403a79209a2575905dd2072e930e398b6d78578f00574732341268`;
konstrukcja i referencje walidacyjne/testowe nie dzielą audytowanych grup dzieł.

Skład 100 dodatnich prac konstrukcyjnych: 53 Post-Impressionism / 47 Realism;
48 sketch-and-study, 20 landscape, 11 genre-painting, 9 portrait, 6 cityscape,
5 still-life i 1 nude-painting. Walidacyjne referencje autora: 7/13 style i
13/20 szkiców; testowe: 9/11 i 8/20. Medium i okres nie są dostępne. Etykieta
landscape lub Post-Impressionism nie oznacza medium olejnego; galeria pokazuje
również rysunki. To kierunek związany z tą konkretną mieszaniną prac autora,
nie ustalony kierunek wyłącznie kolorowego późnego postimpresjonizmu.

## Rzeczywiste wektory i transfer: techniczny N2

Surowy kierunek jest różnicą osobnych średnich dodatnich i ujemnych dzieł,
bez nowego treningu. Dla terminalnego latentu ma 65 536 współczynników
(`4×128×128`); dla obu h warstw 1 280 kanałów po spatial/token mean, następnie
broadcasting offsetu do wszystkich pozycji tego samego modułu. Cztery kanały
latentu nie są RGB. Mid i up to różne przestrzenie, mimo takiej samej liczby kanałów.
LoRA działa tylko w inwersji; probing i cała generacja używają modelu bazowego.

Wyniki na identycznych 2+2 referencjach i głównych negatywach:

| Klucz | Cosinus DDIM/Moonwalker | Kąt | RMS d DDIM / MW | Wspólny s_ref |
|---|---:|---:|---:|---:|
| z_T | 0,987261 | 9,155° | 0,934915 / 0,971941 | 0,953620 |
| z_t:30 | 0,988242 | 8,795° | 0,903164 / 0,930271 | 0,917712 |
| h:mid:30 | 0,853773 | 31,376° | 1,163448 / 1,042361 | 4,948378 |
| h:up:30 | 0,761207 | 40,429° | 0,008024 / 0,008290 | 0,071623 |

Cosinus porównuje wyłącznie tę samą reprezentację/warstwę/stan. Surowe różnice
mają różne normy, ale relative-RMS daje `RMS(delta)=abs(alpha)*s_ref`, wspólnie
dla obu metod. Przykład z_T α0,05: RMS dodatku 0,047681 i L2 około 12,2064.
α steeringu nie jest alpha_cumprod schedulera. Stan30 to post-step inwersji
t581 / alpha_cumprod 0,17986345291137695, nie trzydzieste wywołanie denoisingu.

Duża zgodność DDIM/MW na tych samych obrazach **nie oznacza stabilności autora**:
z_T split-half DDIM 0,000349–0,000887, MW -0,000793–0,001366. Każda połowa
N2 to jedno dodatnie i jedno ujemne dzieło, a trzy powtórzenia podziału nie
stanowią niezależnych zbiorów. W tym małym zestawie główna permutacja etykiet
zamieniła klasy miejscami i daje `-d`, cosinus -1; osobne diagnostyczne
permutacje nie zastępują tej kontroli użytej do generacji. N100 ma obliczyć
split-half, podpróby i losowe/etykietowe kontrole przed transferami.

Pierwszy rzeczywisty z_T α0,05 transfer wykonano dla zdjęcia
`coco2017_val_000000171611`, prompt treści „A man and a dog on a boat in the water.”
Zmiana obejmuje rozmycie/teksturę poziomą i deformację łodzi/człowieka. Kolumny
dwóch metod wyznaczania kierunku są wizualnie bardzo podobne przy tej samej
inwersji źródła. To potwierdza wykonanie offsetu i pokazuje praktyczne ograniczenie,
nie potwierdza stylu van Gogha ani przewagi Moonwalkera.

| Zdjęcie (alpha0, własna rekonstrukcja) | PSNR DDIM / MW [dB] | SSIM DDIM / MW |
|---|---:|---:|
| COCO 171611 | 29,05146 / 25,75429 | 0,858242 / 0,841227 |
| COCO 545730 | 25,82713 / 27,34611 | 0,786296 / 0,807087 |

Kolejność odwraca się między dwoma zdjęciami. Nie licz tych wyników wielokrotnie
dla kolejnych kluczy/α jako nowych próbek ani nie wyciągaj wniosku populacyjnego.
N2 CSD/DINO nie liczono; niezależny sanity CSD dla przyszłego pilota przeszedł:
10+10 dzieł walidacyjnych kontra rozłączne prototypy 10+10, średni margines
dodatnich 0,10913956, ujemnych -0,01145837. Testowe referencje nie uczestniczyły
w sanity. To bramka poprawności evaluatora, a nie pomiar jakości transferu.

## Wznowienie GPU i artefakty świeżego wykonania

Pierwszy smoke v7/v2 **3361803** zakończył się błędem 21:39 CEST przed generacją
edycji: `_use_default_values` Diffusers pochodzi z setu i kolejność metadanych
zmieniała hash pomiędzy procesami. Naprawa sortuje wyłącznie tę listę domyślnych
kluczy; wszystkie parametry naukowe oraz znaczące kolejności list pozostają ścisłe.
Runtime kontraktu v2 wymaga świeżej ekstrakcji, więc nie nadpisano starego cache.
Dodano wymagany hash-bound receipt indexu kierunków. Nieudane v2 zachowano,
pilot/test 3361804/3361805 anulowano bez alokacji.

Obecny immutable snapshot `code_20261012_v8`: 225 plików, source.sha256 SHA256
`3de4ae762aba6bd8c354a6506634ca52f545b2a45e2903a7cbdb14b6615fb222`.
Koordynator PID 3922361 uruchomiono 22:16:44 CEST, stan/logi w
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/autoflow_20261012_v3`.
Ponownie wykorzystano wyłącznie niezależnie sprawdzone manifesty, bo moduł danych
i zamrożone prompty mają identyczne bajty między v7/v8. GPU ekstrakcja, kierunki
i kontrakt powstały od nowa. Smoke 3361924 zakończony; pilot/test zgłoszone
z afterok, bez zmian checkpointu, modułów, CFG, liczebności lub budżetu sweepu.

Lokalne rzeczywiste artefakty: `reports/technical/wikiart_steering_actual_smoke_20261012_v2/`.
`vector_diagnostics_v3`, `evaluation_v3`, `photo_preview_v3` i
`explainer_actual_transfer_v3.html` wiążą świeży kontrakt
`5b89e06fe3c6791cfb3d052fc5d866862bc1bcc1ad41dac6ffd5cb2c45dd88ff`.
Stare `vector_diagnostics`/`explainer.html` opisują wyłącznie historyczną
ekstrakcję v2. Każdy obraz/mapa ma oryginalny hash; mapy używają wspólnych skal
koloru i pokazują surowe współczynniki, nie zdekodowany styl. PCA jest wyłącznie
deskryptorem wizualizacyjnym konstrukcyjnych prac, nigdy dodatkiem do SDXL.
Po zmianie serializacji, indexreceipt i dodaniu diagnostyki całość 162 testów
przeszła w 28,90 s; [receipt](../../reports/technical/wikiart_steering_auto_20261012_v1/final_v3_tests.xml).
Regression obejmuje różne PYTHONHASHSEED w osobnych procesach i zachowuje
odrzucenie rzeczywistej zmiany parametrów schedulera. Ruff/diff-check poprawne.
Zmierzony smoke `cost_v3.json`: mediana referencji 16,7073 s (obie inwersje
i probing, bez VAE encode/ładowania), mediana generacji 5,9343 s (z decode),
peak VRAM 11 305 461 760 bajtów, około 10,53 GiB. Całe zadanie 26:52 obejmuje
też ładowanie, sanity i publikację. Resource limits 2/12/4 h są limitami,
nie zmierzonym czasem N100/testu.

[Surowe wektory NumPy](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/vector_arrays_v3/raw_vectors.npz)
zawierają osiem niezmienionych float32 tensorów N2; plik można odczytać
`numpy.load(path, allow_pickle=False)`. Nazwy: `z_T__ddim`, `z_T__moonwalker`,
`z_t_30__ddim`, `z_t_30__moonwalker`, `h_mid_30__ddim`, `h_mid_30__moonwalker`,
`h_up_30__ddim`, `h_up_30__moonwalker`.
[CSV kanałów](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/vector_arrays_v3/activation_channels.csv)
ma 5 120 wartości z kluczem/metodą/indeksem kanału;
[metadane](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/vector_arrays_v3/provenance.json)
wiążą oryginalne kierunki, kształt, prompt, czas, RMS, backbone/checkpoint i hashe.
Eksport sprawdził 176 plików wejściowych oraz dokładną równość NPZ/CSV po
ponownym odczycie. Nie normalizowano tych zapisanych surowych wartości.

[Niezależny audyt ukończonego smoke](../../reports/technical/wikiart_steering_actual_smoke_20261012_v2/smoke_v3_independent_audit.json)
ma `status=passed`, SHA256
`e0825835c9d24636a74c9be77f1d2b440413dcc4da10c17f30179c05d6b864c1`.
Sprawdzono 19 842 unikalnych hash-bound plików, w tym 18 984 pliki runtime,
wagi modelu/adaptera/evaluatorów, snapshot i publikacje. 249 znaczników
artefaktów i finalny rehash 738 plików wyników/snapshotu przeszły bez zmiany.
Inventory smoke: 722 pliki / 382 546 498 bajtów, 206 PNG edycji, 104 siatki,
43 PNG / 43 PDF diagnostyki. Odtworzono 51 stanów harmonogramu, 36 kierunków
raw/permuted/subpool i wspólne RMS dla 18 kluczy, 108 statystyk wektorów,
246 wierszy (164 photo / 82 seed) z 4 źródeł oraz 84 agregaty/kontrasty bootstrapu.
Odtworzenie metryk pikseli/średnich/CI ma błąd dokładnie 0; cosinus do 1,11e-15,
kąt do 3,03e-7 stopnia przy acos blisko 1. Brak wywołań modeli. Audyt potwierdza
spójność techniczną, nie przewagę stylu lub zachowania treści.

23:03 CEST: pilot opublikował 16/300 referencyjnych tensorów z receiptami,
bez błędu. Uruchomiony osobno read-only observer wizualizacji, PID **4066317**,
automatycznie tworzy `pilot_geometry.html`, `pilot_validation.html` i
`pilot_test.html` po publikacji odpowiednich kompletnych raportów. Output:
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/reports/interactive_pilot_v3`;
atomowy stan/log/launch receipt w tym katalogu. Poll 30 s, bez arbitralnego
limitu oczekiwania kolejki; pojedyncza blokada procesu. Nie ładuje modeli,
nie zgłasza zadań ani nie zmienia kontraktów/selectora. Renderer sprawdza
wszystkie declared image hashes i ten sam kontrakt co wektory. Źródła:
`scripts/observe_wikiart_steering_html.py` oraz
`scripts/render_wikiart_steering_explainer.py`; zdalne przypięte pliki v1/v4
mają SHA256 `61c64d6b…c531` / `05e6d000…b398`. Testy etapowej publikacji,
blokady pojedynczego procesu, timeout/błędu i terminal state oraz JS defaults,
baseline tekstowego i N100 wording przeszły. Bieżący smoke HTML ma 8 386 808
bajtów, domyślny z_T α0,05 i pokazuje wyłącznie istniejące warianty.

Stan początkowy 2026-10-12 18:42 CEST: implementacja i testy CPU, bez wyniku naukowego SDXL.
Użytkownik wskazał Athenę jako preferowany klaster i potwierdził, że dane nie
są przygotowane. Nie zgłoszono treningu ani kosztownego pilota. Test obrazowy
SDXL wymaga przygotowania WikiArt i zdjęć. CSD/DINOv2 są jawnie `unavailable`
w domyślnych konfiguracjach, do czasu audytu wag i preprocessingu.

**Aktualizacja 19:32 CEST:** na kolejne jawne zlecenie użytkownika pobrano cały
WikiArt i uruchomiono automatyczny dalszy przebieg na Athenie. Dane obejmują
81 444 obrazy i 1 889 etykiet van Gogha przed deduplikacją i filtracją. Wszystkie
72 shardy mają zgodne źródłowe SHA256; manifest źródła i metadane są gotowe.
Pobranie oraz eksport zakończyły się o 19:05 CEST. Gotowe są 24 zdjęcia COCO
(12 walidacyjnych i 12 testowych) i osobny runtime z oficjalnymi CSD/DINO,
które przeszły ścisłe ładowanie wag i testy numeryczne.
Koordynator aktualnie audytuje i dzieli źródło; następujące po tym zadania
smoke/pilot/test są automatyczne. Brak jeszcze wyniku naukowego SDXL.

Ścieżka danych:
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/wikiart_d559852_20261012_v1/export/{source,metadata}.json`.
Zdjęcia: `.../wikiart-steering/photos24_20261012_v1/photos.jsonl`;
runtime: `.../wikiart-steering/runtime_20261012_v1/bin/python`.
Automatyzacja: `scripts/wikiart_steering_workflow.py coordinate --snapshot <snapshot>`;
żywy stan/logi/konfiguracje: `.../wikiart-steering/autoflow_20261012_v1`.
Proces 3151255 rozpoczął się o 19:18:41 CEST; kod v5 ma 222 przypięte pliki,
SHA256 inwentarza `6fdcdfeb471427538692d5db8dc4766ce7fc769d97592f1291bd5b0543ddc391`.
V4 był lokalnym pakietem przed domknięciem koordynatora, bez wykonania GPU.

Kolejne kroki, już autoryzowane i zakodowane:

1. Audyt pełnego eksportu, SHA/pHash i grup dzieł; przygotowanie N=100 oraz
   20 referencji walidacyjnych i 20 testowych na grupę. Negatywy główne są
   dopasowane według genre, losowe służą jako kontrola. Unknown Artist jest wykluczony.
2. Smoke z 2+2 referencjami i 2 zdjęciami, utworzony wyłącznie z konstrukcyjnej
   puli pilota. Wydzielone prace walidacyjne sprawdzają evaluator; test pozostaje
   poza doborem parametrów. Limit zadania po audycie kosztu: 2 h.
3. Pilot: inwersja, probing bazowego U-Neta i kierunki; 12 wspólnych seedów oraz
   12 zdjęć walidacyjnych, trzy ustalone klucze (`z_T`, `h:mid:30`, `h:up:30`),
   wspólny sweep alpha i cztery ramiona. Automatyczny wspólny wybór klucza,
   warstwy i czasu. Limit po audycie kosztu: 12 h.
4. Po zaliczeniu poprzedników: 12 niezależnych zdjęć testowych i wybrana
   konfiguracja, pełny wspólny sweep, CSD/DINO, bootstrap po obrazach źródłowych
   i siatki. Limit po audycie kosztu: 4 h.

Reguła wyboru zamrożona przed wynikami: margines CSD minus własne alpha=0 daje
osiągnięty wzrost stylu. Wyznaczamy wspólny dodatni zakres dla wszystkich trzech
kluczy i obu metod: 21 punktów, bez ekstrapolacji. Maksymalizujemy średnie
podobieństwo treści DINO, równo ważąc metody, z identycznym budżetem i pełnym sweepem.
Brak dodatniego zakresu oznacza `no_selection`, bez wymuszonego zwycięzcy.
Ujemne alpha pozostają diagnostyczne. Nie trenuje się nowego adaptera ani nie
skaluje automatycznie do N=200 lub powtórzeń GPU. Wcześniejsze zatrzymanie
przed pilotem został zastąpiony nowym jawnym zleceniem automatycznego wykonania.

Finalny raport weryfikacji evaluatorów (`a6b155…5017`, 17:20:27 UTC) wiąże
aktualny kod metrics (`76f1a908…e832`). Naprawiono ograniczoną deserializację
oficjalnych metadanych CSD i nieobsługiwany argument HF DINO, z testami i bez
zmiany wag, preprocessingu lub ekstrakcji CLS. Sanity autora wymaga prawdziwych
prac walidacyjnych; powtarzalne cechy sztucznych obrazów nie są walidacją jakości stylu.
Lokalne receipts: `reports/technical/wikiart_steering_auto_20261012_v1`.

**Aktualizacja zasobów, 20:09 CEST:** przed zgłoszeniem GPU przygotowano snapshot
`code_20261012_v6` (223 pliki, SHA256 inwentarza
`734286a6e2f1c9c5d29d16493a785c1c5d8ceb5f7d918fa36d8c5e02a9e0f83c`).
Pełny budżet to 10 849 / 114 001 / 20 400 wywołań U-Neta dla smoke/pilota/testu,
z podwojonym batchem CFG; pilot generuje 1 584 unikalne PNG. Historyczny
pomiar Athena 39 600 wywołań w 1:56:44 używał jednej gałęzi conditional, więc
nie daje dopasowanej estymaty tego eksperymentu (źródło i raport:
`docs/experiments/sdxl_watermark_window_20261011.md`). Limity 2/12/4 h obejmują
rezerwę dla dwóch gałęzi, ładowania modeli, VAE i scoringu; nie są zmierzonym
czasem tego biegu. Nowe wyniki wymagają około 10 GB przestrzeni poza danymi
i wagami. Rezerwę czasu zmieniono bez zmiany liczebności, kluczy lub alpha.
Przejście z v5 po kompletnym CPU-manifeście zapisuje
`autoflow_20261012_v1/resource_handoff.json`; przygotowana selekcja jest ponownie
walidowana, a wykonanie GPU będzie pochodziło z v6. Na 20:09 CEST trwa jeszcze
przygotowanie manifestu; nie ma ID zadań GPU ani wyniku jakości.

**Wykryta jakość danych, 20:14 CEST:** pierwszy `prepare` z v5 zatrzymał się
przed wyborem kohort i przed GPU. `wikiart_018465` oraz `wikiart_063028` mają
identyczne SHA256 `b51c3579174001676f4565887cd3a217a871b68d44ec40c93e76d4f153eed1e7`,
lecz przypięte ClassLabels podają odpowiednio `edgar-degas` i `mary-cassatt`.
Nie ma podstaw do arbitralnego rozstrzygnięcia autora. Przejście v5→v6 również
zatrzymało się, zachowując porażkę w `autoflow_20261012_v1`.
Przygotowywana poprawka utrzymuje domyślne odrzucenie sprzecznego źródła
(`conflict_policy=error`), a koordynator jawnie wybiera `quarantine`: wyklucza
całą tranzytywną grupę z konstrukcji i obu zestawów referencji, zapisując
oryginalne etykiety i członków w audycie. Filtr nie korzysta z wyniku żadnego
evaluatora. Wznowienie będzie pochodziło z nowego snapshotu i nowego katalogu
przebiegu; nie ma jeszcze wyliczonych wektorów ani obrazów SDXL.

**Aktualny przebieg, 20:28 CEST:** poprawka została wdrożona i pełny zestaw
149 testów przeszedł w 12,13 s. Weryfikuje m.in. kwarantannę całej grupy,
identyczny wynik serial/4 wątki, niezależne odtworzenie mapowania etykiet i SHA
wszystkich plików przy ponownym użyciu cache. Domyślny tryb `error` zachowano;
koordynator jawnie zapisuje `quarantine` w konfiguracji i manifeście.
Cache pełnych SHA/pHash zapisuje się atomowo przed grupowaniem, więc następna
kontrola etykiet nie wymaga ponownego dekodowania całości.

Wznowienie rzeczywiście uruchomiono o 20:28:02 CEST, PID **3423391**, z
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/code_20261012_v7`.
Snapshot ma 223 pliki; SHA256 inwentarza
`316500f63afd6a70ab68fe902c421fe908bd78762681009596a0a0c89b907c71`.
Żywy stan i logi: `.../wikiart-steering/autoflow_20261012_v2/`.
Pierwszy status to `running/audit_complete_export`; następujące po nim
prepare → smoke → pilot → wybór walidacyjny → test pozostają automatyczne.
Na ten moment nie ma ID zadań GPU ani wyniku naukowego. Nieudane v1 i
nieuruchomiony handoff v6 pozostają zachowane jako historyczna proweniencja.
O 20:31 CEST potwierdzono `running/prepare_pilot_once`, żywy PID 3423391,
bez błędu i bez ID GPU. Przygotowanie wykorzystuje jawną kwarantannę i 4 wątki.

Komenda koordynatora na login node Atheny (istniejącego procesu nie dublować):

```bash
base=/net/tscratch/people/plgmichalsadowski/wikiart-steering
PYTHONPATH="$base/code_20261012_v7" "$base/runtime_20261012_v1/bin/python" \
  "$base/code_20261012_v7/scripts/wikiart_steering_workflow.py" coordinate \
  --snapshot "$base/code_20261012_v7" \
  --workflow "$base/autoflow_20261012_v2" --poll-seconds 30
```

## Zamrożony punkt startowy

- SDXL Base, rewizja `462165984030d82259a11f4367a4eed129e94a7b`, 1024×1024;
  wspólny oryginalny VAE, FP32 encode/decode, skalowanie `0.13025`, UNet FP16.
- DDIM 50 inwersja / 50 generacja, CFG 1/1, epsilon, leading, offset 1,
  `set_alpha_to_one=false`, eta 0. Odzyskany endpoint ma czas 981.
- Jeden historyczny StyleBench-BLIP2 conditional adjacent-epsilon step2500:
  SHA256 `3c9d2bf56c483aa1963f9638ce72e042f50a7879347353ece441f1b80b5b69a0`,
  rank16/alpha8, `to_q,to_k,to_v,to_out.0`, bez FF; 1120 tensorów.
  Hash, moduły, rank, kompletność A/B i rzeczywiste wczytanie adaptera są sprawdzane.
  To osobny eksperyment od Recap attention+FF i SD2.1/FARI; żadna z ich wag
  ani konfiguracji nie jest zastępowana.
- LoRA ON tylko podczas 50 kroków inwersji Moonwalkera. Feature probing,
  denoising, rekonstrukcja, edycja i tekstowy baseline mają LoRA OFF.
- Pomiar A: dwa kierunki na identycznych nowych promptach i znanych seedach,
  bez inwersji zdjęcia. Pomiar B: pełne cztery kombinacje metody referencji
  i metody inwersji zdjęcia, sparowane po obrazie/promptach/siłach.

2026-10-12 odczyt przez SSH potwierdził na Athenie model, checkpoint i ich
konfiguracje. Dostępny interpreter:
`/net/tscratch/people/plgmichalsadowski/vfi/runtime_20261006_v1/bin/python`:
Torch 2.11.0+cu128, Diffusers 0.37.1, PEFT 0.19.1.
W przeszukanym katalogu danych projektu nie znaleziono WikiArt ani jego manifestu.
Istniejące 40 referencji StyleID nie zastępuje zbioru z audytem etykiet autorów.
W sprawdzonym runtime VFI brakuje modułu `clip`, wymaganego przez helper CSD;
domyślne wagi CSD/DINO nie były dostępne w sprawdzonych ścieżkach cache.
To dodatkowa jawna zależność przed naukową ewaluacją, poza przygotowaniem danych.

## Implementacja i dane

Główne wejście: `python -m diff_inversion.eval.wikiart_steering`.
Osobne etapy: `preflight`, `prepare`, `extract`, `build-directions`, `edit`,
`evaluate`, `select`, `cost`. Konfiguracje w `config/experiments/`:
`wikiart_steering_smoke.yaml`, `wikiart_steering_pilot.yaml`,
`wikiart_steering_large.yaml`; korzystają z jawnego `extends` i nadpisań `key=value`.

Lokalny JSON/JSONL WikiArt zawiera `id,path,artist,style,genre`, opcjonalnie
`artwork_group`, `title`, `year`, `period`, `medium`. Nazwy lub indeksy klas
są odczytywane przez osobne `metadata.json`, np.:

```json
{"features":{"artist":{"names":["Claude Monet","Vincent van Gogh","Unknown Artist"]},
             "style":{"names":["Impressionism","Post-Impressionism"]},
             "genre":{"names":["landscape","portrait"]}}}
```

Nie używamy stałego numerycznego ID Van Gogha. Unknown Artist, brakujące
etykiety i konflikty etykiet w grupie reprodukcji nie tworzą pewnych negatywów.
SHA256, pHash i podana tożsamość dzieła łączą warianty przed podziałem;
to audyt heurystyczny, bez gwarancji rozpoznania wszystkich reprodukcji.
Manifest zapisuje ograniczenia, wszystkie grupy, rzeczywiste liczebności,
rozkłady genre/style i dostępne metadane medium/period. Brakujące informacje
nie pozwalają twierdzić, że zidentyfikowano wszystkie szkice lub okresy autora.

Negatywy główne mają dokładny histogram genre pozytywów i limit autora;
losowe negatywy są osobną pulą konstrukcyjną. Referencje validation/test
są rezerwowane wcześniej. Podpróby N=25/50/100/200 są zagnieżdżone i wspólne
dla metod; niedostateczna pula zatrzymuje przygotowanie. Pilot n100 potrzebuje
140 różnych dodatnich dzieł i 240 znanych negatywów przy val20/test20,
z wystarczającym genre/author balance. To minima, nie potwierdzona dostępność.

Zdjęcia: osobny JSONL, np.
`{"id":"photo_val_001","path":"/ABS/photo.jpg","split":"validation","prompt":"A bicycle beside a wall","seed":101}`.
Wymagane osobne testowe zdjęcia, brak duplikatów i opis wyłącznie treści.
Pilot przewiduje 12 validation + 12 test, większy bieg wymaga nowego manifestu
100+ zdjęć. Dostarczono 12+12 niezależnych promptów i rozłączne seedy
w `manifests/wikiart_steering/content_prompts_v1.jsonl`.
Walidator odrzuca jawne nazwy autorów/ruchów; semantyczne przecieki opisów
nadal wymagają przeglądu. Nie wybiera się dzieł według wyniku evaluatora.

## Reprezentacje i ingerencja

`z_T`: cały tensor, dodanie raz przed pierwszym krokiem generacji.
`z_t:k`: cały odzyskany stan, dodanie raz do bieżącej trajektorii denoisingu.
`h:mid:k` / `h:up:k`: pooled kanały bazowego UNet na odzyskanym stanie,
z broadcastingiem w dokładnie tym samym module i czasie generacji.
Moduły są odkrywane przez rzeczywisty forward: mid-block i wyjście `attn2`
w upsampling. Ich nazwy, kształty i konwencje BCHW/BTC są częścią kontraktu.
Pełne mapy nie są zapisywane. Długie okna ingerencji nie są włączone.

Indeksy `[1,10,20,30,40,50]` rosną IMAGE→NOISE. Dla domyślnego schedulera
odpowiadają stanom t=1/181/381/581/781/981. UNet czasu pierwszego kroku
inwersji dostaje t=1, chociaż wejściem jest czysty endpoint ze skalą alpha[0].
Kontrakt niezależnie zapisuje iteration, direction, UNet t, before/after state t,
alpha i pre/post-step. Zduplikowane poziomy i off-by-one zatrzymują wykonanie.
Helper uruchamia podwojony batch CFG; odczyt i hooki używają tylko conditional.

Kierunek to osobne średnie FP32, bez N² par. Raw tensor, norm/RMS, liczebności,
prompt, warstwa/czas i diagnostyka są zapisane. `relative_rms` używa jednego
wspólnego skalara z pozytywów i głównych negatywów obu metod w construction;
dla aktywacji RMS pochodzi z tensorów przed poolingiem. Formuła:
`alpha * s_ref * d / max(RMS(d), epsilon)`.
`raw_mean_difference` jest diagnostyczny; prawie zerowe kierunki mają
zapisany stan i nie stanowią dowodu jakości. Skala nigdy nie pochodzi ze zdjęcia testowego.

Zapisano split-half, nested-N, permutacje oraz 3 identycznie losowane
półpule referencji. `edit --repeat 0` używa całej puli;
`--repeat 1/2/3` bada zmienność od półpuli (N50 dla pilota N100), nie trzy
niezależne wektory z N100 ani nowe treningi. Każde powtórzenie uruchamia się jawnie.
Kontrole: alpha=0 obu metod, przeciwny kierunek przez ujemne alpha, losowy
i permutowany kierunek o dopasowanej amplitudzie, wspólny kierunek w obu
inwersjach zdjęcia oraz osobny tekstowy baseline.

Manifest, checkpoint, dokładne model/VAE bytes, kod, wersje zależności,
preprocessing, prompty, LoRA schedule, scheduler, czasy i warstwy wiążą cache.
Hashy artefaktów i completion receipts nie zastępuje samo istnienie pliku.
Hooki, scheduler oraz stany PEFT są przywracane także po błędzie.

## Komendy od danych do raportu

Wybierz Python3.11 z zależnościami projektu; normalnie `uv sync --frozen`.
Lokalne `.venv/bin/python` ma stare dowiązanie snap/264. Testy wykonano przez
działający Python snap/current z `PYTHONPATH=.venv/lib/python3.11/site-packages:$PWD`,
bez zmiany istniejącego środowiska.

Opcjonalny import HF: pełna rewizja, cache-only; pobieranie dopiero przez jawne
`--allow-download`. Nie uruchamia się obcego skryptu datasetu.
Wydana rewizja ma 72 shardy Parquet i 81 444 obrazy, około 33,7GB samych
shardów; cache Arrow i kopie obrazów wymagają dodatkowego miejsca.
Przy pierwszym wdrożeniu nie pobrano datasetu; późniejszy autoryzowany download
zakończył się19:05CEST, jak opisano w aktualizacji powyżej.

```bash
python -m diff_inversion.data.wikiart_steering export-hf \
  --revision d559852d2b232e0fcf195e775866964f0564f2b5 \
  --output-dir /ABS/wikiart --cache-dir /ABS/hf-cache --allow-download
export STEERING_WIKIART=/ABS/wikiart/source.json
export STEERING_LABELS=/ABS/wikiart/metadata.json
export STEERING_DATA_REVISION=d559852d2b232e0fcf195e775866964f0564f2b5
export STEERING_PHOTOS=/ABS/photos.jsonl
CFG=config/experiments/wikiart_steering_smoke.yaml
python -m diff_inversion.eval.wikiart_steering preflight --config "$CFG"
python -m diff_inversion.eval.wikiart_steering prepare --config "$CFG"
python -m diff_inversion.eval.wikiart_steering extract --config "$CFG"
python -m diff_inversion.eval.wikiart_steering build-directions --config "$CFG"
python -m diff_inversion.eval.wikiart_steering edit --config "$CFG" --task seed
python -m diff_inversion.eval.wikiart_steering edit --config "$CFG" --task photo
python -m diff_inversion.eval.wikiart_steering evaluate --config "$CFG"
python -m diff_inversion.eval.wikiart_steering cost --config "$CFG"
```

Lokalna ścieżka z gotowym manifestem pomija `export-hf`. Smoke inwertuje tylko
2+2 główne referencje; zapisany manifest dodatkowo rezerwuje heldout works.
Pilot jest osobnym rootem/configiem i obejmuje również losowe negatywy;
`edit.negative_variants=[negative_matched,negative_random]` ustala kontrolę
przed ekstrakcją, a nie po zobaczeniu testu. Dla N25/50 zmień `data.n` i output.

Evaluator włącz przez jawne, zweryfikowane pola z
`manifests/wikiart_steering/evaluator.example.json`, przed ekstrakcją i
zamrożeniem kontraktu. Sam fakt posiadania pliku CSD nie wystarcza: loader
sprawdza checkpoint/helper/preprocessing i niezależny sanity check na
validation artwork. DINO ma lokalną przypiętą rewizję i komplet hashów.
LPIPS jest opcjonalnym przypiętym eksportem TorchScript.

Po przeglądzie pełnych krzywych walidacyjnych zamroź wspólną konfigurację;
wybór wymaga dostępnego CSD/DINO i mierzonego wspólnego zakresu stylu:

```bash
python -m diff_inversion.eval.wikiart_steering select --config "$CFG" --key z_T
python -m diff_inversion.eval.wikiart_steering edit --config "$CFG" --task photo \
  --split test --selection reports/results/wikiart_steering/smoke_v1/selection.json
python -m diff_inversion.eval.wikiart_steering evaluate --config "$CFG" --split test
```

Nie wykonuj wyboru naukowego na smoke. To przykład składni; właściwy wybór
przeprowadza się na pilocie. Test odrzuca zmienione dowody walidacyjne.
Nie wybieramy osobnych sił/metod o różnych budżetach; porównujemy cały
wspólny zmierzony alpha sweep bez ekstrapolacji.

Athena: zbuduj nowy source-only snapshot i przenieś go do nowego katalogu:

```bash
python scripts/package_wikiart_steering.py --output /tmp/wikiart-steering-snapshot
# Przenieś snapshot na Athenę, zachowując source.sha256, i ustaw poniższe ścieżki.
export STEERING_SNAPSHOT=/ABS/athena/wikiart-steering-snapshot
export STEERING_OUTPUT=/ABS/athena/wikiart-steering-smoke
bash "$STEERING_SNAPSHOT/scripts/launch_wikiart_steering_athena.sh" dry-run
bash "$STEERING_SNAPSHOT/scripts/launch_wikiart_steering_athena.sh" preflight
bash "$STEERING_SNAPSHOT/scripts/launch_wikiart_steering_athena.sh" submit
```

Launcher zgłasza tylko bounded smoke: A100, 8 CPU, 64GB, limit 1h,
bez requeue/nested sbatch i automatycznego pilota. Brak źródeł zatrzymuje
preflight przed `sbatch`. Dane i wagi pozostają poza snapshotem.
`cost.json` podaje dokładną liczbę forwardów, zmierzony czas/VRAM i estymatę
większego biegu, jeśli istnieje prawdziwy pomiar GPU. Obecnie throughput
SDXL jest niezmierzony — czas toy UNet nie jest jego estymatą.

## Artefakty i interpretacja

Powstają `manifest.json`, `contract.json`, atomowe cache/markery,
`directions/*.pt` z kompatybilnością, edit completion receipts,
`per_image.csv/json`, agregaty, sparowane bootstrapy po źródłowych obrazach,
zmienność od półpul, krzywe styl–treść PNG/PDF oraz jawne i zakodowane siatki.
Klucz kodów jest osobnym plikiem. CSD prototypy: L2 każdy obraz → średnia
w grupie → L2, niezależne val/test i validation sanity. DINO porównuje edycję
z oryginałem i rekonstrukcją alpha0. PSNR/SSIM/LPIPS dotyczą przede wszystkim alpha0.
Nie ma FID/ArtFID na małej próbie ani twierdzenia o jakości na podstawie normy d.

Wykonana demonstracja techniczna:
`reports/technical/wikiart_steering_cpu_20261012_v2` (poprzedni v1 zachowany), odtwarzana przez
`python scripts/wikiart_steering_cpu_smoke.py --output /ABS/fresh/toy-output`.
Używa fikcyjnych obrazów/etykiet, prostego VAE/UNet i prawdziwych schedulerów,
bez SDXL/checkpointu/GPU. Zapisała 160 wierszy i 48 siatek; CSD/DINO/LPIPS
pozostały unavailable. Wyniki nie stanowią oceny stylu ani przewagi Moonwalkera.
Testy obejmują realny mały Diffusers UNet i mapowanie schedulerów, wspólne VAE
latents, dokładne alpha0, single-time live-state steering, conditional branch,
sprzątanie po błędzie, hash/module tampering, dzieła/splity i bootstrap.
109 testów CPU przeszło lokalnie i na Athenie; XML są zapisane w `tests.xml`
i `athena/tests.xml`. Dodatkowy test uruchamia rzeczywiste etapy
build-directions → photo/seed edit → evaluate na syntetycznych zasobach,
z kompletną publikacją oraz odrzuceniem brakującego upstream markeru.
Ruff, składnia bash i kontrola whitespace przeszły dla nowych plików.

Finalny source-only snapshot na Athenie:
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/code_20261012_v3`;
207 przypiętych plików, SHA256 inventory
`73e43a25b7cffbeb3724c35b172f1640937b9ef61db8b7ec10e7091f3f6f0e35`.
Testy wykonano w v2, a wszystkie pliki Python w v3 są identyczne bajtowo;
jedyna różnica to CPU przygotowanie/audyt danych przed `sbatch` w launcherze.
Oba snapshoty zachowano. `athena/snapshot.json` dokumentuje zgodność.
`athena/preflight.json` potwierdza model/checkpoint i brak source/labels/photos.
GPU jobs: brak; żadnego naukowego pomiaru WikiArt nie wykonano.

Sukces przyszłego pilota: poprawa treści przy porównywalnym osiągniętym stylu
lub stylu przy porównywalnej treści, w wspólnym zmierzonym zakresie i z
niepewnością po obrazach oraz referencjach. Lepsza rekonstrukcja/normalność
szumu nie gwarantuje liniowej edytowalności. Brak wyniku SDXL oznacza brak
wniosku o przewadze którejkolwiek metody.

## Przeczytane źródła i granice adaptacji

[Haas et al., §V i Appendix F](https://arxiv.org/html/2303.11073v2)
motywują różnicę średnich i kierunki z inwersji realnych obrazów; pełne
przestrzenne h-space w ich pracy nie jest naszym kanałowym poolingiem.
[Kod semantic-diffusion](https://github.com/renhaa/semantic-diffusion) ma MIT.
[CASteer](https://arxiv.org/html/2503.09630v4), w tym Appendix I.3, opisuje
kierunki cross-attention i DDIM style transfer. Jego projekcyjne erasure i
różne operacje nie zastępują naszej jawnej addytywnej różnicy średnich.
[Kod CASteer](https://github.com/Atmyre/CASteer) ma MIT.
[Through Van Gogh’s Eyes](https://lakeeye1220.github.io/Global_Style_Transfer/)
uczy offset stylu; tego treningu tutaj nie odtwarzamy. Nie skopiowano
implementacji tych publikacji; nowe hooki podlegają testom SDXL.

[WikiArt HF](https://huggingface.co/datasets/huggan/wikiart) zawiera osobne
ClassLabel artist/style/genre. Rewizja powyżej jest pinem opcjonalnego eksportu,
nie dowodem pobrania danych. [CSD](https://github.com/learn2phoenix/CSD)
i [artykuł](https://arxiv.org/html/2404.01292v1) dostarczają evaluator stylu;
README ostrzega o rozbieżnościach wydanych wag z artykułem. Kod ma MIT,
[oficjalne wagi](https://huggingface.co/tomg-group-umd/CSD-ViT-L) CC-BY-4.0.
[DINOv2](https://github.com/facebookresearch/dinov2) ma Apache-2.0 dla
standardowego kodu i wag. Wagi/preprocessing trzeba sprawdzić przed oceną;
nie zastępujemy niedostępnego CSD innym modelem pod tą nazwą.
# Odczyt okien aktywacji i następna próba — 2026-10-13 14:20 CEST

Nowy smoke3362255 zakończył się0:0 w23:49. Pilot3362256 policzył całą macierz
1,764 ocen na12znanych seedach i12fotografiach, po czym o13:50CEST zatrzymał
się na `no_selection`. Kod1:0 wSlurm wynika z zatrzymania selekcji; generowanie
i ocenaCSD/DINO zakończyły się poprawnie. Test3362257 został anulowany przed
startem. Wynik dotyczy dodawania osobnego, zgodnego czasowo wektora w każdym
z pierwszych5/10kroków; nie jest wynikiem wcześniejszej edycjizT.

| Okno cross-attention | Maksymalny średni gainCSD DDIM | Maksymalny średni gainCSD Moonwalker |
|---|---:|---:|
| first5 | +0.0002031 | +0.0003478 |
| first10 | +0.0004445 | 0 |

To obserwacje known-seed, dodatnia siatkaλ0..0.2, gain względem własnegoλ0.
Przyλ0.2 wszystkie cztery gain są ujemne; DINO pozostaje0.99958–0.99975.
Zmiany są małe, niemonotoniczne i porównywalne z kontrolami. Tekstowy baseline
ma gain+0.081188 iDINO0.601643. Wspólny dodatni zakres selekcji wynosi[0,0],
ponieważ MWfirst10 nie ma dodatniego gain. Midfirst10 pozostaje kontrolą i nie
blokuje selekcji. Nie wykazano użytecznego transferu ani przewagiMoonwalkera.
CLIP nie został policzony w tym przebiegu. Pełne wyniki, obrazy i HTML:
`reports/results/wikiart_steering/activation_window_20261013_v1/`.

CosineDDIM/MW tej samej warstwy/czasu wynosi0.8340–0.9490 dla up41..50.
Jednak split-halfup50 dlaMW to0.209/−0.289/−0.001, a dlaDDIM
0.364/0.072/0.461. Podobieństwo kierunków między metodami nie wystarcza do
wniosku o stabilnym ani użytecznym stylu; podpul to nie niezależne treningi.

Użytkownik upoważnił dalsze ograniczone eksperymenty bez kolejnych pytań.
Przygotowywana oddzielna próba czułości zwiększaλ do0.5/1/2 przy tych samych
up5/up10. Wspólna siatka[-0.5,0,0.2,0.5,1,2], random/permuted przyλ1,
baseline tekstowy, wszystkie12fotografii walidacyjnych i12known seeds,
cztery sparowane photoarms. To adaptacyjny eksperyment rozwojowy po odczycie
pilota; nie nowy niezależny test ani zmiana starego kryterium sukcesu.
Nie uruchamia selektora ani testu. Ocenia pełną macierz1,188wierszy i zapisujeHTML.

Stanowiska konstrukcyjne zostaną przeniesione jako niezależne kopie dokładnych
bajtów; jawnecfg.lineage, oryginalneSHA i receipt opisują pochodzenie każdego
cache. Stare markery i pliki pozostają niezmienione. Świeży runtime sprawdza
model/checkpoint/moduły/scheduler/branch/dtype. Rebuild kierunków musi odtworzyć
każdy raw/random/permuted/subpool tensor i s_ref dokładnie, poza nowym hashem
kontraktu. Jedyny nowy odczyt konstrukcyjny to dummyprobe warstwy; brak ponownej
inwersji obrazówWikiArt. Budżet55,203 nowych wywołańUNet obejmuje3runtimeprobes;
39,000 konstrukcyjnych wywołań jest reuse. Limit jednejA100 to3h. Worker
automatycznie policzyCSD/DINO, sprawdzi publikację i wyrenderujeHTML.

2026-10-13 14:24CEST: mocniejszeλ faktycznie zgłoszono o14:23:25 jako
Athena **3363029**, RUNNINGt0020/derive_reference_cache. Snapshot
`code_20261013_lambda_v1` ma233przypiętepliki; planSHA256
`747c7acfa136ab87b3e11862d41849c86b62d397fb2f5b11bd124fc99ffb5831`.
44testy lokalne13.14s/Athena24.97s iRuffPASS. Workflow
`wikiart-steering/followup_lambda_20261013_v1`; przyszłyHTML
`run/lambda_validation.html`. Nie ma jeszcze wyników większychλ.

14:26:53CEST: importGPU zakończył się poprawnie (300niezależnych kopii cache,
jawnylineage, parentunchanged). Nowy fingerprintkontraktu
`b13c2a84557637f7e9a3d77f5ed7a00fb661f2a0e6491e87e0c57e47b24cb66b`;
workerprzeszedł dobuild_directions. Niezależnyreplay staregoN100window z200
positive/matchedcache odtworzył40rawh tensors,120split-halfcosines i20s_ref
zresidual0. Nie znaleziono błędu zapisu różnicyśrednich/skali/diagnostyk.

## 2026-10-13 — przyszła ablacja warstw i całego odszumiania

Użytkownik poprosił o zapisanie kolejnego eksperymentu: porównanie steeringu
w wielu warstwach oraz podczas **wszystkich 50 wywołań generacyjnego UNet**,
obok obecnych pierwszych 5/10. To plan przyszłych badań; nie uruchomiono go
i nie zmienia on zakończonego pilota ani bieżącej próby większych λ. Wcześniej
należy odczytać sparowane porównanie rekonstrukcji DDIM/Moonwalker na tych
samych obrazach, wskazane przez użytkownika jako test zastosowania adaptera.
Lepsza rekonstrukcja byłaby użyteczną przesłanką, lecz sama nie dowodzi jakości
kierunku stylu.

Najbliższy precedens to [CASteer v4, §3.1, Appendix H.5 i I.3](https://arxiv.org/html/2503.09630v4).
Autorzy stosują osobne kierunki we wszystkich wyjściach cross-attention.
W ablacji SDXL pojedyncza warstwa dawała znikomy efekt usuwania złości;
wiele warstw mogło zmieniać układ i tożsamość obrazu. Dla usuwania Van Gogha
ostatnie 36 CA uzyskało ten sam Acc_e=0.00 co wszystkie CA, z niższym
LPIPS_u=0.12 zamiast 0.19. To usuwanie stylu z kierunkami ze sparowanych
promptów generacyjnych, nie nasz addytywny steering z inwersji WikiArt.
Ich pokazany transfer realnych zdjęć wykorzystuje DDIM i sterowane odszumianie
po częściowej inwersji; „intensity” oznacza jej głębokość, nie nasze λ ani K.

[SAeUron, §5.1 oraz Appendix C/H](https://arxiv.org/html/2501.18052)
wybiera blok stylu up.1.2 przez ablację bloków SD1.5 dostrojonego do
UnlearnCanvas i pokazuje dodatnie wzmacnianie cech SAE w każdym czasie.
[Oficjalny kod](https://github.com/cywinski/SAeUron) identyfikuje hook jako
`unet.up_blocks.1.attentions.2`, czyli cały blok, nie nasze pojedyncze attn2.
[Haas et al., §III/V](https://arxiv.org/html/2303.11073v2) używają pełnych
przestrzennych map bottleneck w kolejnych czasach. Wyniki tych prac uzasadniają
konkretne kandydatury; żaden hook ani pokrycie czasów nie gwarantuje działania
naszych kanałowych średnich w SDXL przy stałym tekście `A painting`.

Prospektywny zakres porównania:

| Oś | Kandydaci | Co zamrozić przed generacją |
|---|---|---|
| Warstwy | obecne pojedyncze attn2; późna grupa CA; wszystkie CA | pełne nazwy, kolejność rzeczywistych wywołań i kształty wyjść |
| Czas | pierwsze 5; pierwsze 10; środkowe okno; późne okno; pełne 50 | dokładne listy wywołań, stanów i czasów z `state_map.denoising_calls` |

„Wszystkie” początkowo oznacza **wszystkie moduły cross-attention**, nie
jednoczesne dodawanie do CA, self-attention, MLP i konwolucji. Późna grupa
odnosi się do miejsca w sieci, a późne okno do czasu odszumiania. Wariant
ostatnich 36 CA jest kandydatem inspirowanym CASteer; jego faktyczne pokrycie
naszego SDXL trzeba potwierdzić enumeracją modułów i próbnym forwardem.
Najpierw porównujemy pokrycie warstw przy jednym zamrożonym oknie, następnie
pokrycie czasu dla ograniczonego zestawu miejsc, zamiast jednoczesnego
dużego przeszukiwania wszystkich kombinacji.

Dla każdej pary `(warstwa, czas)` konstruujemy od nowa jej własne
`d[method,layer,t]` i konstrukcyjne `s_ref[layer,t]`, wspólne dla DDIM/Moonwalkera.
Identyczna liczba kanałów nie oznacza identycznej bazy ani możliwości użycia
wektora z innej warstwy. Pełne 50 wymaga nowych odczytów brakujących stanów;
obecny cache okna 41–50 nie pokrywa ich wszystkich. Nie powielamy jednego
kierunku na 50 czasów ani nie mieszamy cache różnych kontraktów. Czytnik
i generacja nadal używają bazowego UNet z LoRA OFF; zachowujemy checkpoint,
VAE, prompty, zdjęcia/seedy, CFG1, 50/50 NFE oraz cztery ramiona zdjęć.

Rozdzielamy liczbę miejsc od siły całej ingerencji. Raportujemy wariant
ze stałą λ w każdej warstwie/czasie oraz kontrolę o zamrożonym budżecie
`E = Σ_(layer,time) [RMS(delta[layer,time]) / s_ref[layer,time]]²`.
Dla L miejsc i K czasów z równymi wagami, ustawienie
`lambda[layer,time] = A / sqrt(L*K)` daje E=A². To suma lokalnych
względnych RMS, nie równa energia pikseli ani równy efekt przyczynowy.
Zapisujemy również kształty aktywacji i odpowiedź epsilon, aby sprawdzić,
czy zmiana naprawdę przechodzi dalej przez UNet.

Nowy protokół i budżet GPU trzeba zamrozić przed jego wykonaniem. Pierwsza
ocena pozostaje walidacyjna z λ0, losowymi/permutowanymi/przeciwnymi
kierunkami, baseline tekstowym, CSD/DINO i sparowanymi siatkami obrazów.
Parametry środkowego/późnego okna oraz grupy CA są na razie nieustalone.
Nie dobieramy ich po oglądaniu niezależnego testu i nie zmieniamy historycznej
reguły `no_selection`. Obserwowana poprawa po zwiększeniu liczby warstw lub
czasów byłaby nowym wynikiem, nie reinterpretacją obecnego braku transferu.

## 2026-10-13 — autoryzowane rozszerzenie danych, warstw i czasu generacji

Użytkownik zlecił uruchomienie większej próby i kolejnych zadań diagnostycznych,
aby oddzielić zbyt małą pulę konstrukcyjną od słabego kierunku oraz zbyt małego
pokrycia interwencji. To nowa autoryzacja ograniczonego **N200** i poniższego
grafu; zastępuje dawny brak autoryzacji N200 w tym konkretnym zakresie.
Nie oznacza zgody na nieograniczone powtórzenia, N400 ani nowy trening.
Ta sekcja zamraża prospektywny plan. Job IDs i faktyczne przejścia bramek
należy dopisać po zgłoszeniu; sam opis nie jest potwierdzeniem uruchomienia.

Główny endpoint to **generacja nowego obrazu ze wspólnego szumu**, następnie
pomiar, czy dodawanie kierunku czyni go bardziej podobnym do stylu Van Gogha.
Każdy kontrast używa tego samego rzeczywistego `x_T` wygenerowanego historyczną
konwencją: generator CPU i szum float32, następnie oryginalny cast do FP16
dla UNet, przy tym samym promcie treści i schedulerze. Pierwsze 12 jednostek
zachowuje prompty i szum rodzica: seed `42000+i`, gdzie `i=0,...,11`.
Kolejne trzy powtórzenia dodają do tego seeda odpowiednio 100, 200 i 300,
dając 36 nowych szumów. Stan początkowy, seed i jego hash muszą być
utrwalone w kontrakcie. W tym grafie nie inwertujemy fotografii źródłowych.
Inwersje dzieł służą konstrukcji kierunków; porównanie ich jakości pozostaje
oddzielną diagnostyką. Zachowanie inicjalnego szumu i promptu nie deklaruje
zgodności pikseli nowych obrazów sterowanych nowymi kierunkami z historycznymi
obrazami; publikuje się własne wyjścia i ich hashe.

Model, native VAE, rozdzielczość 1024, CFG1 i DDIM50 pozostają zamrożone.
Adapter pozostaje **StyleBench-BLIP2 conditional adjacent epsilon,
attention-only r16/alpha8, step2500**, SHA256
`3c9d2bf56c483aa1963f9638ce72e042f50a7879347353ece441f1b80b5b69a0`;
moduły `to_q`, `to_k`, `to_v`, `to_out.0`, bez FF. LoRA działa wyłącznie
podczas inwersji referencji. Czytnik wszystkich aktywacji oraz generacja
pracują z LoRA OFF. Nie uczymy adaptera, generatora ani SAE.

### Odczyty, zagnieżdżenie danych i diagnostyka kierunków

Nowa pula ma **200 Van Gogha + 200 dopasowanych innych autorów**. Konstrukcja
N100 jest dokładnym, zamrożonym prefiksem każdej puli N200, z tymi samymi
wcześniejszymi dziełami, a nie niezależnym losowaniem. Zachowujemy rozłączne
20+20 dzieł walidacyjnych oraz 20+20 testowych. Test nie uczestniczy w nowej
konstrukcji, diagnostyce ani wyborze parametrów. Losowe negatywne dzieła
pozostają metadanymi manifestu; nie trafiają do GPU nowego collectora ani do
średnich/skali. Kontrole losowego i permutowanego wektora pozostają aktywne.

Collector odczytuje **wszystkie 50 stanów i wszystkie rzeczywiste wyjścia CA**
w pojedynczym forwardzie każdego czasu, osobno dla obu metod inwersji.
Odczyty przestrzenne dają własny pooled wektor i unpooled RMS każdej pary
`(layer,t)`; nie kopiujemy kierunku między warstwami/czasami. Pełne nazwy,
kształty i kolejność wykonania CA wymagają audytu runtime. „Ostatnie 36 CA”
oznacza końcowe 36 w audytowanym forwardzie, nie końcowe nazwy w porządku
alfabetycznym czy domyślnej enumeracji drzewa modułów.

Koszt collectora: 400 dzieł konstrukcyjnych + 40 walidacyjnych, dwie metody,
50 inwersji + 50 odczytów na dzieło/metodę, oraz jeden runtime probe:
**88 001 nowych wywołań UNet**. Limit: jedna A100, 8 h, 64 GB RAM.
Po ekstrakcji kierunki buduje CPU w **tej samej alokacji GPU**, z zerem
wywołań modelu. Limit 8 h obejmuje obie fazy, a etap CPU ma dodatkowy
timeout 1 h / 64 GB; przekroczenie jest technicznym zatrzymaniem grafu.
Audyt `sinfo` potwierdził partycje GPU Atheny, więc nie zgłaszamy osobnego
zadania CPU ani nie zakładamy niezweryfikowanej partycji/konta CPU.
Wspólne `s_ref[layer,t]` pochodzi wyłącznie z **całej konstrukcji N200,
obu klas i obu metod**; tę samą skalę stosujemy dla N100/N200 i DDIM/MW.
Walidacja nie uczestniczy w `d` ani `s_ref`.

CPU raportuje split-half i stabilność kierunków oraz rozdzielenie autora na
rozłącznych dziełach walidacyjnych, m.in. AUC i Cohen's d projekcji. Raportuje
osobno całą pulę, `sketch_and_study` i pozostałe dzieła. To diagnostyka sygnału,
nie warunek dowodzący, że jego dodanie do nowej trajektorii zmieni styl.
Większa pula może obniżyć wariancję średniej; nie naprawia automatycznie
niedopasowanej reprezentacji lub sprzecznych składników mieszaniny.

### Dwa zadania generacyjne

Pierwsze zadanie utrzymuje **N100, 12 dotychczasowych promptów i jeden seed
na prompt**. Zmienia pokrycie warstw/czasów przy wspólnej dawce:

| Grupa | Liczba warstw L | Wywołania K |
|---|---:|---:|
| `up10` | obecne pojedyncze CA | pierwsze 10 |
| `up50` | obecne pojedyncze CA | wszystkie 50 |
| `last36CA:first10` | 36 końcowych CA | pierwsze 10 |
| `allCA:first10` | wszystkie CA | pierwsze 10 |
| `allCA:full50` | wszystkie CA | wszystkie 50 |

Każda grupa ma obie metody kierunku oraz pięć przypadków: kierunek przy
`lambda = [-1, 0.2, 1]`, losowy wektor przy 1 i permutowany przy 1.
Przeciwny znak −1 jest kontrolą kierunku, nie poszukiwaniem optymalnej
stylizacji przeciwnym offsetem. Dodatkowa diagnostyka `allCA:full50` ma
**stałe lambda_eff = 0.2 w każdej warstwie/czasie**, obie metody. Każdy source
ma jedną wspólną generację lambda0 i baseline tekstowy. Łącznie **648 obrazów,
32 401 wywołań UNet**, limit jedna A100 / 4 h.

Drugie zadanie porównuje **N100 z N200** na 48 jednostkach generacyjnych:
12 tych samych promptów × 4 zamrożone seedy. Obie metody i te same pięć
przypadków obejmują grupy `up50` oraz `allCA:full50`, przy wspólnej dawce.
Po jednej bazie lambda0 i tekście na source daje to **2016 obrazów,
100 801 wywołań UNet**, limit jedna A100 / 8 h. To oddziela rozmiar konstrukcji
od pokrycia i precyzji ewaluacji. Cztery seedy promptu nie stają się czterema
niezależnymi promptami w przedziałach ufności.

Dla porównania dawki w L warstwach i K czasach:

```text
lambda_eff = lambda * sqrt(10 / (L*K))
sum_(layer,time) (RMS(delta) / s_ref)^2 = 10 * lambda^2
```

Punktem odniesienia jest dotychczasowe pojedyncze CA przez 10 wywołań.
Równa suma względnych RMS nie oznacza równego efektu na epsilon, percepcję
czy piksele. Wariant stałego 0.2 jest osobną diagnostyką znacznie większej
łącznej dawki i nie uczestniczy w porównaniu dawki kontrolowanej.

### Ocena, granice i graf wykonania

Ocena używa dotychczasowych przypiętych CSD/DINO, prototypów i sanity autora.
Główne wyniki to gain marginesu CSD względem własnej wspólnej lambda0 oraz
DINO względem tej bazy; tekstowy prompt stylu pozostaje kontrolą dodatnią.
Porównania są sparowane po identycznym source/prompt/seed. Bootstrap losuje
całe **klastry promptów**, zachowując wszystkie seedy i ramiona tego promptu.
Pokazujemy pełne siatki i kontrole. CLIP w tym grafie nie liczymy; możliwe
uzupełnienia oceny opisuje [nota matematyczna](wikiart_steering_mathematical_theory_20261013.md).
Nie zmieniamy historycznego selektora, nie promujemy testu i nie wykonujemy
automatycznego wyboru najlepszej konfiguracji.

Graf ma **trzy zadania GPU**: pierwsze wykonuje extraction i CPU directions
w jednej alokacji, a dopiero po ich poprawnym zakończeniu startują coverage
oraz size, dopuszczone do równoległego wykonania. Maksimum dwie A100
równocześnie; extraction ma jedną. Łączny prospektywny koszt wynosi
**221 203 nowych wywołań UNet**. Dla każdego etapu wymagamy przypiętych
wejść, niezmiennego snapshotu, completion receipts i fail-closed weryfikacji
rodzica przed uruchomieniem zależnego zadania. Żaden etap nie trenuje modelu.
Według stanu przygotowania o 17:42 CEST trwa pełny audyt SHA/pHash oparty
na zapisanym audycie rekordów i weryfikacji bajtów; nowy graf nie miał jeszcze
GPU Job IDs. To status przygotowania, nie wynik nowych odczytów lub generacji.

### Wyniki dostępne przed rozszerzeniem

Stronger-lambda **3363029** zakończył się `0:0` 2026-10-13 o 17:10:56 CEST,
publikując 1188 wierszy oceny. Na 12 known-seed wszystkie niezerowe badane
lambda mają ujemny średni gain dla up5/up10 i obu metod. Przy lambda2 gain
up5 DDIM/MW to −0.0009000/−0.0011637, a up10 −0.0018639/−0.0023785;
DINO do bazy pozostaje 0.9952–0.9968. Tekstowy baseline ma +0.08118786
przy DINO 0.60164318. Większa siła samego dotychczasowego kierunku nie
rozwiązała braku stylizacji w tym protokole. Małe dodatnie wtórne gains na
fotografiach po inwersji MW nie zmieniają tego głównego odczytu.
[Archiwum i ograniczenia readback](../../reports/results/wikiart_steering/lambda_sensitivity_20261013_v1/README.md)
potwierdzają hashe pobranych raportów i replay średnich; pełne oryginalne
PNG/tensory/scoring modelowy nie były niezależnie odtwarzane lokalnie.

Diagnostyka rekonstrukcji ośmiu stałych dzieł wskazuje średnie PSNR
21.75555/22.04174 dB i DINO 0.601107/0.684388 dla DDIM/MW. MW poprawia
te średnie, ale znaczące niepowodzenia rekonstrukcji występują w obu metodach;
latent MSE poprawia się tylko na 2/8 dzieł. To mała próba opisowa, nie bramka
populacyjnej użyteczności adaptera. Sugeruje problem dopasowania inwersji do
realnych dzieł jako osobną hipotezę; nie ustanawia przyczyny słabego steeringu.
[Galeria i sparowane wyniki](../../reports/technical/wikiart_steering_visual_review_20261013_v1/artwork_roundtrip_8/gallery.html).

### 2026-10-13 18:22:23 CEST — rzeczywista publikacja grafu do kolejki

Zgłoszenie na Athenie zakończyło się poprawnie: extraction+CPU-directions
**3363566**, coverage **3363567**, size **3363568**. Oba dzieci mają własne
`afterok:3363566`, więc mogą działać niezależnie po ukończeniu ekstrakcji
i zapieczętowaniu kierunków. Ten wpis potwierdza submission; nie stwierdza
jeszcze RUNNING, ukończenia nowych obrazów ani wyniku jakościowego.

Workflow: `wikiart-steering/diagnostics_expanded_20261013_v1`.
Niezmienny snapshot wykonawczy:
`/net/tscratch/people/plgmichalsadowski/wikiart-steering/code_20261013_expanded_v1`.
Inventory obejmuje **240 przypiętych plików**, SHA256
`d0c235196f81d09da596ed0470c6ab10aea4277abf285ec1583830848be3c44b`.
Plan SHA256:
`7e5d5e8a7367cbf52acaa5f33ea3e1b9c16fc5fc00d7b5d57e535e00cb9d4a8d`.
Submission receipt SHA256:
`e7c5a3bc4e760926f7eaacf4a5a14e711892ea8fb020326ec155bc6ee0e28d57`.

Przed zgłoszeniem wszystkie **163 testy** przeszły lokalnie w **18.29 s**
i na Athenie w **22.97 s**, Ruff przeszedł oraz sprawdzono **18 984**
przypięte pliki odziedziczonego runtime. Zachowano wyżej zamrożone
221 203 wywołania UNet, limity 8/4/8 h, maksymalnie dwie A100, historyczny
CPU float32 szum rzutowany do FP16 oraz brak treningu/selekcji/test use.
Ta lokalna aktualizacja po zgłoszeniu nie zmienia uruchamianego snapshotu,
jego prelaunch dokumentacji ani zapieczętowanego planu.

**18:23:27 CEST — rzeczywisty odczyt Slurma:** extraction+directions
3363566 jest `RUNNING` na `t0012`, elapsed `1:04`. Coverage3363567
i size3363568 pozostają `PENDING (Dependency)`. Stan workera3363566:
`running/extract`, z aktualizacją18:22:25 CEST. Ten odczyt potwierdza start
ekstrakcji, nie ukończenie cache ani wynik jakościowy.


## 2026-10-13 18:43 CEST — bieżący stan rozszerzonego grafu i pobieranie

3363566 nadal działa na t0012 (21:10); 3363567/3363568 oczekują. Runtime
potwierdził 70 wyjść CA, oryginalne up to ca_034. Do 18:35 collector zapisał
21 receipts z dokładnym odtworzeniem historycznych stanów/odczytów/RMS.
Nie jest to jeszcze ukończony kierunek N200 ani wynik jakości stylizacji.
Niezależny audyt uruchomienia przeszedł 27 kontroli; potwierdza zagnieżdżone
pule, holdouty, 240 plików snapshotu, zasoby/zależności i oba wyniki 163 testów.
Ograniczenia sprawdzenia wejść dostępnych tylko zdalnie zapisano w
`reports/technical/wikiart_steering_expanded_20261013_v1/launch_independent_audit.json`.

Pierwszy lokalny obserwator 1227491 przerwał pobieranie po błędzie DNS
o 18:30:16; źródło i receipts zachowano w technicznym `failed_observer_v1/`.
Naprawiony obserwator ruszył o 18:42:32, PID 1238762, trwała sesja 91550,
limit 24 h. Heartbeaty 18:42:33 i 18:43:33 są `waiting`, bez błędu transportu.
SHA256 źródła: `14dac00cc1c1b71fb2b513931ae2563f5795b63c2235854e3636c250b51126e4`.
16 testów i Ruff przeszły. Ponawia SSH255/timeouts co 60 s w tym samym
limicie; błąd hashów/formatu kończy odczyt. Nie zmienia jobów ani danych
na Athenie. Lokalny cel to
`reports/results/wikiart_steering/expanded_diagnostics_20261013_v2/`.
Po ukończeniu zbierze sprawdzone raporty, 192 PNG z pierwszych dwóch
ustalonych źródeł generacji i 32 przykłady original/VAE; odtworzy macierz
przypadków oraz średnie/kontrasty, bez ponownej oceny modelowej, losowań
bootstrapu lub pobierania tensorów. Aktualna strona pokazuje oczekiwanie.
Notatki i obserwator nie zmieniają zamrożonego GPU snapshotu/planów.

## 2026-10-14 rano — ukończone N200, coverage i size

Graf zakończył się w pierwotnym budżecie: **3363566 `0:0`**, koniec
13.10 o22:26:56 CEST, elapsed4:04:33; **3363567 `0:0`**, koniec
13.10 o23:42:34, elapsed1:15:38; **3363568 `0:0`**, koniec14.10
o02:40:29, elapsed4:13:33. Rzeczywiste liczniki wynoszą odpowiednio
88 001,32 401,100 801, razem **221 203 wywołania UNet**. Nie zmieniono
checkpointu, attention-only modułów, VAE, CFG1 ani50-NFE generacji.

Collector objął440 dzieł. Dla200 historycznych rodziców odtworzył dokładnie
6000 tripletów stan/up-feature/unpooled-RMS. CPU builder odtworzył także
raw/random/permuted/RMS z30 historycznych plików kierunków.70 rzeczywistych
CA, oryginalne up `ca_034`, dało14 000 geometrycznych i42 000 projekcyjnych
wierszy; generation/scoring opublikował648 obrazów coverage i2016 size.
To potwierdza wykonanie kontraktu, a wynik jakości trzeba oceniać oddzielnie.

### Wynik głównego endpointu: brak wykazanej użytecznej stylizacji

W coverage wszystkie22 warunki dodatniego kierunku mają ujemny średni gain
marginesu CSD. Zwiększenie liczby warstw i czasu nie dało potwierdzonej
użytecznej stylizacji ani ogólnej przewagi kierunku Moonwalkera. Szczególnie
duża dawka, stałe0.2 w każdej z70 CA przez50 wywołań, daje DDIM/MW
gain **−0.0070998/−0.0086032** i DINO **0.87249/0.86765**. Ta kontrola
wyraźniej zmienia treść, lecz nie poprawia średniej miary stylu; pozostaje
osobna od ramion o wyrównanej sumie względnych RMS².

W size, na12 promptach ×4 seedach, pojedyncze up/full50/lambda1 ma gains
N100 DDIM/MW **+0.0002515/+0.0001887**, a N200
**−0.00005254/+0.00010658**. Dla N200/all70/full50/lambda1 gains wynoszą
**−0.00073061/−0.00111516**, DINO **0.97785/0.97643**. Tekstowy styl na
tych samych48 jednostkach ma gain **+0.0656156**, DINO **0.626263**.
Wszystkie12 sparowanych nominalnych95% przedziałów CSD dla N200−N100,
włącznie z kontrolą znaku−1, obejmuje zero. Nie wykryto korzyści większej
konstrukcji w tych warunkach; nie oznacza to równoważności metod,
uniwersalnej nieskuteczności większych zbiorów ani wyniku niezależnego testu.

### Co pokazuje sama geometria kierunków

W dotychczasowym up:first10 średni split-half rośnie N100→N200:
DDIM **0.3883→0.6949**, MW **0.1646→0.6015**. Walidacyjne AUC
zmienia się równocześnie DDIM **0.5550→0.54275**, MW **0.5470→0.5100**.
Na tym samym module ostatnie10 wywołań generacji ma N200 AUC
**0.8105/0.7785**. Większa pula stabilizuje estymację, lecz nie naprawia
słabej separacji autora w dawnym wczesnym oknie. Późniejszy sygnał jest
hipotezą do osobnej interwencji, nie dowodem, że późne wstrzyknięcie zadziała.
Wiersze warstwa/czas i trzy podziały split-half współdzielą te same dane.
Nie traktujemy14 000/42 000 komórek jak niezależnych próbek. Analiza nie
odtwarzała indywidualnych AUC/cosinusów z surowych cache.

### Pobranie i niezależne odtworzenie wyników

Drugi obserwator przestał działać bez receipt zakończenia, po ostatnim
heartbeat13.10 o23:17:48 CEST. Nie ma podstaw do przypisania mu konkretnej
przyczyny; to inna historia niż jawny błąd DNS pierwszego obserwatora.
Ręczny readback ukończył się14.10 o06:42:34 CEST, a dodatkowe podglądy
o06:45:32. Potwierdził hashe656 zadeklarowanych zdalnych artefaktów coverage
oraz2024 size; lokalnie są raporty i504 oryginalne PNG:192 wcześniej
ustalone generacje,280 dodatkowych z wszystkich12 tematów/r0 i32 original/VAE.
Pozostałe oryginalne obrazy oraz duże cache pozostają na Athenie.

Niezależny replay96 grup agregatów ×3 metryki i66 kontrastów ×2 obejmuje
średnie i2000-losowaniowe CI bootstrapu po klastrach promptów, z maksymalnym
błędem **3.33e−16**.264 wspólne deklaracje hashów PNG coverage/size są
identyczne. Nie odtwarzano pierwotnych wyjść encoderów CSD/DINO. Miara CSD
pozostaje uczonym proxy, a przedziały są nominalne, nie skorygowane za cały
zbiór eksperymentów. Odczyt nie wykonuje nowych wywołań modeli, treningu
ani zgłoszeń GPU i nie zmienia historycznej selekcji lub testu.

Archiwum główne: `reports/results/wikiart_steering/expanded_analysis_20261014_v1/`.
[Receipt ręcznego readback](../../reports/results/wikiart_steering/expanded_analysis_20261014_v1/manual_readback_completion.json),
[metryki, kontrole i ograniczenia](../../reports/results/wikiart_steering/expanded_analysis_20261014_v1/metric_analysis/README.md),
[pełna diagnostyka kierunków](../../reports/results/wikiart_steering/expanded_analysis_20261014_v1/direction_analysis/READOUT.md).
Oryginalne pobrane publikacje są w
`reports/results/wikiart_steering/expanded_diagnostics_20261013_v2/`.

### Dalsze propozycje, bez uruchomienia

Można osobno porównać ostatnie10 wywołań z wyrównaną dawką i zamrozić
nową konstrukcyjną pulę bez etykiety sketch-and-study. Etykieta gatunku nie
certyfikuje medium; dobór wymaga prospektywnego kontraktu przed generacją.
Nie uruchomiono tych prób, nie wybrano automatycznie konfiguracji,
nie zmieniono progu jakości i nie promowano niezależnego testu. Obecne
wyniki zachowują negatywną interpretację w swoim dokładnym zakresie.

## 2026-10-14 09:52 CEST — dodatkowy pomiar przyczynowy aktywacji

Użytkownik ponownie sprecyzował cel: dodatek do aktywacji w kolejnych krokach,
a nie przesunięcie szumu startowego. Odczyt kodu i wcześniejszych receipts
potwierdza, że ukończone expanded już realizowało first10/full50 w aktywacjach.
Nowa analiza job3363157 sprawdziła330 zapisanych kontroli dodatku i przebieg
50kroków, bez nowych wywołań modeli; nie wykazała resetu ani pominięcia hooka.
Raport: `reports/results/wikiart_steering/activation_delivery_20261014_v1/`.

Na prośbę o dalsze zbadanie działania zgłoszono odrębny job3365402:
N200/up/ca_034, dwa pierwsze ustalone prompty walidacyjne, DDIM/MW,
kierunek i random, matched lambda1. Okna1–10,21–30,41–50,1–50.
Każde aktywne wywołanie porównuje dwa rzeczywiste forwardy na tym samym
bieżącym latencie, a osobno mierzy nagromadzoną różnicę względem bazy.
36 obrazów,2441UNet calls, jedna A100 do45min, bez nowej ekstrakcji,
treningu, wyboru parametrów i testu. Późne okno z wcześniejszej propozycji
jest teraz częścią zgłoszonej małej diagnostyki, nie potwierdzoną naprawą.

83 testy przeszły lokalnie i na Athenie. Snapshot v3 i plan są zamrożone;
plan SHA256 `32415ac906c473b9c9a15777066880d3618429ced3df2a4d306c8a0c6d6f2e66`.
Zgłoszono09:52:01 CEST, początkowy stan PENDING/Priority. Nie ma jeszcze
ukończonego wyniku nowej diagnostyki. Pierwszy job3365401 zakończył się
przed załadowaniem modelu na błędzie str/Path; zero UNet calls. Naprawa
i test pełnego workera są w nowym snapshotcie, nie w historycznych plikach.

Szczegóły, ograniczenia i historia:
[audyt aktywacji](wikiart_activation_causal_audit_20261014.md).

### Ukończenie dodatkowej diagnostyki, 2026-10-14 10:00 CEST

Job3365402 ukończony0:0 o09:59:29,7:27, dokładnie2441UNet calls i36PNG.
Wszystkie44 artefakty pobrano i sprawdzono;640 zapisanych kontroli dodatku
jest dokładnych po kwantyzacji.12 wspólnych PNG/szumów odtwarza historyczny
size dokładnie. Przed pierwszym hookiem każda trajektoria zgadza się z bazą.
CompletionSHA `230b556ee93cb04d1e925c7a57e50a9bde7cd57cc30a3fbaf99a79928026329b`.

Last10 daje średni gain DDIM/MW+0.00078521/+0.00007788 z DINO
0.9999214/0.9999161; tekst+0.05168869/DINO0.86393091. MW ma przeciwny znak
gainu na rowerze i chacie. Lokalna odpowiedź epsilon wzrasta first10→last10
0.03873%→0.12802%DDIM i0.03765%→0.13279%MW, także w bezwzględnym RMS,
ale random odpowiada silniej. Poprawny dodatek i większa odpowiedź nie dają
tu użytecznego stylu. To dwa ustalone prompty, bez wyboru okna lub wniosku
populacyjnego. Duży N200 zachowuje wcześniejszą negatywną interpretację.

Raport/figura/galeria36 przypadków:
`reports/results/wikiart_steering/activation_causal_trace_analysis_20261014_v1/`.
Lokalnie replayowano54 średnie agregatów i16 średnich kontrastów z błędem0;
nie odtwarzano encoderów, bootstrapowych CI ani prymitywnych tensorów hooków.

## 2026-10-14 — propozycja prostszego testu syntetycznego koloru

Na prośbę użytkownika zapisano odrębny
[plan black dog−white dog](synthetic_color_steering_20261014.md)
z każdą inwersją przy `dog`, zapisanymi prawdziwymi stanami generacji,
kontrolami neutralnego odczytu i lokalnym pomiarem koloru sierści.
To plan, bez nowych jobów lub wyników. Nie zmienia historycznych kontraktów,
selektora ani negatywnego odczytu WikiArt. Syntetyczny styl i rzeczywisty kolor
pozostają dalszymi pomostami do zbadania rozjazdu domen i trudności pojęcia.
