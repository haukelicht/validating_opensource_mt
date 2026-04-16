
# Languages ----

language_iso2c_to_name <- c(
  "bg" = "bulgarian",
  "bs" = "bosnian",
  "ca" = "catalan",
  "cat" = "catalan",
  "cs" = "czech",
  "da" = "danish",
  "dan" = "danish",
  "de" = "german",
  "deu" = "german",
  "el" = "greek",
  "ell" = "greek", 
  "gre" = "greek", 
  "en" = "english",
  "eng" = "english",
  "es" = "spanish",
  "esp" = "spanish",
  "et" = "estonian",
  "fi" = "finish",
  "fi" = "finnish",
  "fin" = "finish",
  "fr" = "french",
  "fra" = "french",
  "gl" = "galician",
  "glg" = "galician",
  "hr" = "croatian",
  "hu" = "hungarian",
  "it" = "italian",
  "ita" = "italian",
  "lt" = "lithuanian",
  "lv" = "latvian",
  "nb" = "norwegian",
  "nl" = "dutch",
  "nld" = "dutch",
  "no" = "norwegian",
  "nob" = "norwegian",
  "pl" = "polish",
  "por" = "portuguese",
  "pt" = "portuguese",
  "ro" = "romanian",
  "ru" = "russian",
  "sk" = "slovak",
  "sl" = "slovenian",
  "spa" = "spanish",
  "sv" = "swedish",
  "swe" = "swedish",
  "tr" = "turkish",
  "uk" = "ukrainian",
  NULL
)


language_iso2c_to_iso3c <- c(
  # mapping of language ISO-2c to ISO-3-c (ISO 639-2, B) codes
  # see, e.g., https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
  "bg" = "bul",
  "bs" = "bos",
  "ca" = "cat",
  "cs" = "ces",
  "cs" = "cze",
  "da" = "dan",
  "da" = "dnk",
  "de" = "deu",
  "de" = "ger",
  "el" = "ell",
  "el" = "gre",
  "en" = "eng",
  "es" = "esp",
  "es" = "spa",
  "et" = "est",
  "fi" = "fin",
  "fr" = "fra",
  "gl" = "glg",
  "hr" = "hrv",
  "hu" = "hun",
  "it" = "ita",
  "lt" = "lit",
  "lv" = "lav",
  "nb" = "nob",
  "nb" = "nor",
  "nl" = "dut",
  "nl" = "nld",
  "no" = "nob",
  "no" = "nor",
  "pl" = "pol",
  "pt" = "por",
  "ro" = "ron",
  "ru" = "rus",
  "sk" = "slk",
  "sl" = "slv",
  "sv" = "swe",
  "tr" = "tur",
  "uk" = "ukr",
  NULL
)

# Machine translation models ----

mt_model_map <- c(
  "deepl" = "DeepL",
  "google" = "Google Translate",
  "google_old" = "Google Translate (old)",
  "opus-mt" = "OPUS-MT",
  "m2m_100_1.2b" = "M2M (1.2B)",
  "m2m_100_418m" = "M2M (418M)",
  "multilingual" = "multilingual"
)

mt_model_type_map <- c(
  "deepl" = "commercial",
  "google" = "commercial",
  "google_old" = "commercial",
  "opus-mt" = "open-source",
  "m2m_100_1.2b" = "open-source",
  "m2m_100_418m" = "open-source",
  "multilingual" = "open-source"
)

mt_model_type_color_map <- c("commercial" = "#56B4E9", "open-source" = "#E69F00")

mt_model_color_map <- c(
  "deepl" = "#944be3",
  "google" = "#5685e9",
  "google_old" = "#56B4E9",
  "opus-mt" = "#e3ab4b",
  "m2m_100_1.2b" = "#a33f05",
  "m2m_100_418m" = "#d55e00",
  "multilingual" = "#b5e34b"
)

# finetuning datasets ----

dataset_map <- c(
  "dupont_and_rachuj_2022" = "Dupont & Rachuj (2022)",
  "lehmann+zobel_2018" = "Lehmann & Zobel (2018)",
  "poljak_2022" = "Poljak (2023)",
  "theocharis_et_al_2016" = "Theocharis et al. (2016)",
  "cmp_translations_sample" = "Manifesto Translations corpus (Ivanusch & Regel, 2024)"
)
