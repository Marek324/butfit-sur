# Zadání projektu do SUR 2025/2026

**Bodové ohodnocení:** 25 bodů

## Úkol

Úkolem je natrénovat detektor jedné osoby z obrázku obličeje a hlasové nahrávky. Trénovací vzory jsou k dispozici v archívu na adrese:

<https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2025-2026/SUR_projekt2025-2026.zip>

## Struktura dat

Tento archív obsahuje adresáře:

- `target_train`
- `target_dev`

kde jsou trénovací vzory pro detekovanou osobu ve formátu **PNG** a **WAV**.

V adresářích:

- `non_target_train`
- `non_target_dev`

jsou potom negativní příklady.

## Omezení dat a augmentace

Pro trénování detektorů je dovoleno použít **pouze tato poskytnutá data**. **NENÍ POVOLENO** jakékoli využití jiných externích řečových či obrázkových dat, jakožto i použití již předtrénovaných modelů (např. pro extrakci reprezentací (embeddings) obličejů nebo hlasu).

Tato data ale můžete *„augmentovat"* tak, že vytvoříte nové trénovací vzory např.:

- přidáním šumu do nahrávek či obrázků,
- rotací, posunutím či změnou velikosti obrázků,
- změnou rychlosti řeči apod.

## Generalizace

Důležitým cílem projektu je **schopnost systému generalizovat** na neznámá data i při použití velmi omezeného množství poskytnutých trénovacích dat. Evaluační data, která budou zveřejněna později (viz níže), budou obsahovat vzory s odlišnými vlastnostmi (např. šum, změny kvality obrazu apod.), které se nemusí přesně vyskytovat v trénovacích datech.

Při návrhu systému proto zvažte:

- vhodné techniky augmentace dat,
- volbu modelu s ohledem na omezené množství dat,
- omezení přetrénování (overfittingu).

Tyto aspekty musí být jasně popsány v dokumentaci (viz níže).

## Baseline systémy

Na přednášce při zadávání projektu si předvedeme jednoduché baseline systémy (**MFCC + GMM** pro audio, **lineární klasifikátor** pro obrázky). Pro dosažení plného bodového hodnocení **nestačí tyto systémy pouze reprodukovat nebo mírně upravit**. Očekává se vlastní přístup reflektující výše uvedené požadavky.

## Validační strategie

Do adresářů `*_train` a `*_dev` byla trénovací data rozdělena především pro účely trénování a testování již zmíněných „baseline" systémů. Z důvodu omezeného rozsahu poskytnutých dat ovšem takové rozdělení nebude optimální a pro dosažení plného bodového hodnocení bude nutné použít vhodnou validační strategii (např. **křížovou validaci** umožňující efektivně trénovat i testovat na všech datech).

Pro tento účel může být užitečné respektovat informace o tom, které trénovací vzory patří stejné osobě a zda byly pořízeny v rámci jednoho nahrávacího sezení. Jméno souboru pro každý vzor je rozděleno do polí pomocí podtržítek:

```
f401_01_f21_i0_0.png
 │    │
 │    └── číslo nahrávacího sezení (01)
 └── identita osoby (f401)
```

Pro trénování doporučujeme použít křížovou validaci respektující strukturu dat (např. podle nahrávacích sezení, tj. aby vzory z jednoho sezení nebyly současně v trénovacích i validačních datech). Zvolený způsob evaluace během vývoje musí být popsán a zdůvodněn v dokumentaci.

## Evaluace

Ostrá evaluační data, na kterých budou vaše systémy vyhodnoceny, budou k dispozici v **neděli 3. května ráno**. Tato data budou obsahovat řádově stovky souborů ke zpracování. Vaším úkolem bude tato data automaticky zpracovat vašimi systémy (věříme Vám, že nebudete podvádět a dívat se na obrázky či poslouchat nahrávky) a nahrát soubory s výsledky do informačního systému (IS).

### Formát souboru s výsledky

ASCII soubor se třemi poli na řádku oddělenými mezerou:

| Pole | Popis |
|------|-------|
| 1 | Jméno segmentu (jméno souboru bez přípony `.wav` či `.png`) |
| 2 | Číselné skóre - čím větší hodnota, tím si je systém jistější, že se jedná o hledanou osobu |
| 3 | Tvrdé rozhodnutí: `1` pro hledanou osobu, jinak `0` (pro apriorní pravděpodobnost 0,5) |

## Požadované systémy

V jakém programovacím jazyce budete implementovat váš detektor či pomocí jakých nástrojů budete data zpracovávat, záleží jen na Vás. Každá skupina musí odevzdat alespoň:

1. **jeden systém založený pouze na obrazových datech**,
2. **jeden systém založený pouze na audio datech**,
3. **jeden systém kombinující obě modality** (multimodální fúze).

Zvolený způsob kombinace modalit (např. na úrovni příznaků nebo skóre) musí být popsán a zdůvodněn v dokumentaci.

## Odevzdání

- Odevzdat můžete i více souborů s výsledky, **maximálně však námi bude zpracováno 6 takových souborů**.
- Soubory s výsledky můžete do **pondělí 4. května 23:59** nahrávat do IS.
- Klíč se správnými odpověďmi bude zveřejněn **5. května**.
- Na poslední přednášce **6. května 2026** bychom měli analyzovat Vaše výsledky a řešení.

### Skupiny a struktura archívu

Na tomto projektu budete pracovat ve skupinách (**1 až 2 lidé**), do kterých se můžete přihlásit v IS.

Jména souborů s výsledky pro jednotlivé systémy volte tak, aby se podle nich dalo poznat, o jaký systém se jedná (např. `audio_GMM`, `image_linear`).

Každá skupina nahraje všechny soubory s výsledky zabalené do jednoho ZIP archívu se jménem `login1_login2.zip` či `login1.zip`, podle toho, kolik Vás bude ve skupině.

Obsah archívu:

```
login1_login2.zip
├── *.txt                 (soubory s výsledky)
├── SRC/                  (zdrojové kódy implementovaných systémů)
└── dokumentace.pdf
```

> **Do ZIP archívu prosím nepřikládejte evaluační data!**

## Dokumentace

Soubor `dokumentace.pdf` bude v českém, slovenském nebo anglickém jazyce popisovat Vaše řešení. Dokumentace (**cca 3 strany A4**) musí obsahovat:

- popis navržených systémů (audio, obraz, multimodální),
- popis použité validační strategie,
- způsob řešení generalizace a omezení přetrénování.

Dokumentace musí umožnit pochopit, **proč** byl systém navržen právě tímto způsobem, nikoli pouze **jak** je implementován. Důraz věnujte úvahám vedoucím k finálnímu návrhu systému: jak jste systémy během jejich vývoje vyhodnocovali a které techniky či rozhodnutí se pozitivně projevily na úspěšnosti systému.

Tento dokument bude také popisovat, jak získat Vaše výsledky pomocí přiloženého kódu. Uveďte:

- jak kódy zkompilovat,
- jak systémy spustit,
- kde hledat výsledné soubory,
- jaké externí nástroje je nutné instalovat a použít.

## Ústní obhajoba

Součástí hodnocení bude **krátká ústní obhajoba** po odevzdání projektu, která proběhne v prvních týdnech zkouškového období (časové sloty budou upřesněny). Studenti by měli být schopni vysvětlit návrh svého řešení a odpovědět na dotazy k použitým metodám.

## Hodnocení

Plný počet bodů (**25**) získá řešení, které:

- správně načítá data a produkuje výsledky,
- obsahuje funkční systémy pro obraz, audio i jejich kombinaci,
- používá vhodnou validační strategii,
- reflektuje problém generalizace a omezeného množství dat,
- obsahuje kvalitní dokumentaci umožňující reprodukci a pochopení,
- je úspěšně prezentováno u ústní obhajoby.

V případě nesplnění některých požadavků bude bodové hodnocení sníženo.

