import csv
import pickle
import re
from flask import Flask, render_template, request

app = Flask(__name__)

ALBANIAN_DATASET = "com-shqip me komente.csv"
ENGLISH_DATASET = "train_data.csv"
LANGUAGE_TEXT = {
    "sq": {
        "page_title": "Analiza e Komenteve Online",
        "header_title": "Tema e Diplomës",
        "header_subtitle": "Analiza e Komenteve Online dhe Klasifikimi i Tyre",
        "header_description": (
            "Sistem inteligjent për analizimin e komenteve dhe klasifikimin e tyre "
            "në pozitive, negative ose neutrale duke përdorur Machine Learning."
        ),
        "language_label": "Gjuha",
        "textarea_label": "Shkruaj komentin për analizë",
        "textarea_placeholder": "Shkruaj komentin këtu...",
        "submit_button": "Analizo Komentin",
        "result_title": "Rezultati i Analizës",
        "result_text": "Komenti është:",
        "stats_count_suffix": "komente",
        "stats_total_label": "Totali i komenteve",
        "stats_note_en": None,
        "stats_title_sq": "Komentet në shqip",
        "stats_title_en": "Komentet në anglisht",
        "info_goal_title": "Qëllimi",
        "info_goal_text": (
            "Të ndërtohet një aplikacion web që analizon komentet tekstuale "
            "dhe përcakton nëse ato kanë sentiment pozitiv, negativ apo neutral."
        ),
        "info_tech_title": "Teknologjitë",
        "info_tech_text": "Python, Flask, HTML, CSS, Machine Learning, Scikit-learn",
        "info_features_title": "Funksionaliteti",
        "info_features_text": (
            "Përdoruesi shkruan një koment dhe sistemi jep menjëherë rezultatin "
            "e analizës së sentimentit, si edhe përqindjet e komenteve në dataset."
        ),
        "labels": {
            0: "Negativ",
            1: "Pozitiv",
            2: "Neutral",
        },
        "stat_labels": {
            "positive": "Pozitive",
            "negative": "Negative",
            "neutral": "Neutrale",
        },
        "language_names": {
            "sq": "Shqip",
            "en": "English",
        },
    },
    "en": {
        "page_title": "Sentiment Analysis of Online Comments",
        "header_title": "Diploma Project",
        "header_subtitle": "Sentiment Analysis and Classification of Online Comments",
        "header_description": (
            "An intelligent system that analyzes comments and classifies them "
            "as positive, negative, or neutral using machine learning."
        ),
        "language_label": "Language",
        "textarea_label": "Enter a comment for analysis",
        "textarea_placeholder": "Write your comment here...",
        "submit_button": "Analyze Comment",
        "result_title": "Analysis Result",
        "result_text": "The comment is:",
        "stats_count_suffix": "comments",
        "stats_total_label": "Total comments",
        "stats_note_en": None,
        "stats_title_sq": "Albanian Dataset",
        "stats_title_en": "English Dataset",
        "info_goal_title": "Goal",
        "info_goal_text": (
            "To build a web application that analyzes textual comments "
            "and determines whether their sentiment is positive, negative, or neutral."
        ),
        "info_tech_title": "Technologies",
        "info_tech_text": "Python, Flask, HTML, CSS, Machine Learning, Scikit-learn",
        "info_features_title": "Functionality",
        "info_features_text": (
            "The user enters a comment and the system immediately returns "
            "the sentiment analysis result together with the dataset distribution."
        ),
        "labels": {
            0: "Negative",
            1: "Positive",
            2: "Neutral",
        },
        "stat_labels": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        },
        "language_names": {
            "sq": "Albanian",
            "en": "English",
        },
    },
}

# Ngarkimi i modelit dhe vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

POSITIVE_WORDS = {
    "awesome", "excellent", "fantastic", "good", "great", "love", "nice",
    "perfect", "super", "wonderful", "amazing", "happy", "best",
    "positive", "pozitiv", "pozitive",
    "mire", "mir", "mirë", "shume mire", "shum mire", "shumë mirë",
    "pelqen", "pëlqen", "dua", "dashuri", "fantastike", "shkelqyer",
    "shkëlqyer", "buker", "bukur", "super", "kendshem", "këndshëm",
    "fantastik", "mrekullueshme", "mrekullueshëm", "jashtezakonshem",
    "jashtëzakonshëm", "lumtur", "kenaqur", "kënaqur",
}

NEGATIVE_WORDS = {
    "awful", "bad", "boring", "hate", "horrible", "poor", "sad",
    "terrible", "ugly", "worst", "disappointing", "problem", "annoying",
    "negative", "negativ", "negative",
    "keq", "shume keq", "shum keq", "shumë keq", "urrej", "tmerr",
    "tmerrshme", "katastrofe", "katastrof", "katastrofë", "neverit",
    "neveri", "i keq", "e keqe", "bosh", "zhgenjyer", "zhgënjyer",
    "merzitur", "mërzitur", "shkaterruar", "shkatërruar", "demtuar",
    "dëmtuar", "vonesa", "papershtatshem", "papërshtatshëm",
    "papershtatshme", "papërshtatshme",
}

NEGATIONS = {"nuk", "s'kam", "ska", "s'ka", "jo", "mos", "sme", "s'me", "smë", "s'më"}

POSITIVE_PHRASES = {
    "me pelqen", "më pëlqen", "shume mire", "shumë mirë", "shume bukur",
    "shumë bukur", "ia vlen", "eshte super", "është super", "shume i mire",
    "shumë i mirë", "shume e mire", "shumë e mirë", "jam i kenaqur",
    "jam i kënaqur", "jam e kenaqur", "jam e kënaqur", "me ka pelqyer",
    "më ka pëlqyer", "teper i mire", "tepër i mirë", "do te kthehem perseri",
    "do të kthehem përsëri", "ishte fantastik", "shume pozitive",
    "shumë pozitive",
}

NEGATIVE_PHRASES = {
    "nuk me pelqen", "nuk më pëlqen", "s me pelqen", "s'me pelqen",
    "s më pëlqen", "s'më pëlqen", "sme pelqen", "smë pëlqen",
    "sme pëlqen", "shume keq", "shumë keq", "fare keq",
    "nuk ia vlen", "nuk eshte i mire", "nuk është i mirë",
    "nuk eshte e mire", "nuk është e mirë", "eshte katastrofe",
    "është katastrofë", "eshte tmerr", "është tmerr", "jam i zhgenjyer",
    "jam i zhgënjyer", "jam e zhgenjyer", "jam e zhgënjyer",
    "ishte i demtuar", "ishte i dëmtuar", "ishte e demtuar",
    "ishte e dëmtuar", "erdhi i demtuar", "erdhi i dëmtuar",
    "erdhi e demtuar", "erdhi e dëmtuar",
}

NEUTRAL_WORDS = {
    "neutral", "neutrale", "mesatare", "normale", "ok", "njesoj", "njësoj",
}

NEUTRAL_PHRASES = {
    "pozitive dhe negative",
    "pozitiv dhe negativ",
    "pa koment",
    "as mire as keq",
    "as mirë as keq",
    "nuk kam mendim",
    "eshte ne rregull",
    "është në rregull",
}


def build_stat_items(counts, total, label_order, stat_labels):
    return [
        {
            "label_key": label_key,
            "label": stat_labels[label_key],
            "count": counts.get(source_label, 0),
            "percent": round(counts.get(source_label, 0) * 100 / total, 2) if total else 0,
            "css": css_class,
        }
        for source_label, label_key, css_class in label_order
    ]


def read_label_counts(path, field_name, allowed_labels, encoding):
    counts = {label: 0 for label in allowed_labels}
    total = 0

    with open(path, newline="", encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = str(row.get(field_name, "")).strip()
            if label in counts:
                counts[label] += 1
                total += 1

    return counts, total


def load_dataset_stats(language):
    text = LANGUAGE_TEXT[language]
    sq_counts, sq_total = read_label_counts(
        ALBANIAN_DATASET,
        "Sentiment",
        ("1", "0", "0/1"),
        "utf-8-sig",
    )
    en_counts, en_total = read_label_counts(
        ENGLISH_DATASET,
        "sentiment",
        ("1", "0", "2", "0/1"),
        "utf-8",
    )

    return {
        "sq": {
            "title": text["stats_title_sq"],
            "total": sq_total,
            "items": build_stat_items(
                sq_counts,
                sq_total,
                (
                    ("1", "positive", "positive"),
                    ("0", "negative", "negative"),
                    ("0/1", "neutral", "neutral"),
                ),
                text["stat_labels"],
            ),
        },
        "en": {
            "title": text["stats_title_en"],
            "total": en_total,
            "items": build_stat_items(
                en_counts,
                en_total,
                (
                    ("1", "positive", "positive"),
                    ("0", "negative", "negative"),
                    ("2", "neutral", "neutral"),
                ),
                text["stat_labels"],
            ),
        },
    }

def normalize_text(text):
    lowered = text.lower()
    cleaned = re.sub(r"[^0-9a-zA-Zçë\s']", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def keyword_sentiment(text):
    normalized = normalize_text(text)
    if not normalized:
        return None

    tokens = normalized.split()
    if len(tokens) < 2:
        return None

    for phrase in NEUTRAL_PHRASES:
        if phrase in normalized:
            return 2

    score = 0

    for phrase in NEGATIVE_PHRASES:
        if phrase in normalized:
            score -= 4
    for phrase in POSITIVE_PHRASES:
        if phrase in normalized:
            score += 3

    for phrase in POSITIVE_WORDS:
        if " " in phrase and phrase in normalized:
            score += 2
    for phrase in NEGATIVE_WORDS:
        if " " in phrase and phrase in normalized:
            score -= 2

    has_neutral = any(token in NEUTRAL_WORDS for token in tokens)
    has_positive = False
    has_negative = False

    for index in range(len(tokens)):
        window_two = " ".join(tokens[index:index + 2])
        window_three = " ".join(tokens[index:index + 3])

        if window_three in POSITIVE_PHRASES or window_two in POSITIVE_PHRASES:
            score += 3
            has_positive = True
        if window_three in NEGATIVE_PHRASES or window_two in NEGATIVE_PHRASES:
            score -= 3
            has_negative = True

    for index, token in enumerate(tokens):
        if token in NEUTRAL_WORDS:
            has_neutral = True

        if token in NEGATIONS:
            window = tokens[index + 1:index + 4]
            if any(candidate in POSITIVE_WORDS for candidate in window):
                score -= 2
                has_negative = True
            if any(candidate in NEGATIVE_WORDS for candidate in window):
                score += 2
                has_positive = True

    if has_neutral or (has_positive and has_negative):
        return 2

    if score >= 2:
        return 1
    if score <= -2:
        return 0
    return None


def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vectorized)[0]
    ml_prediction = int(model.predict(text_vectorized)[0])
    keyword_prediction = keyword_sentiment(text)
    confidence = float(max(probabilities))

    if keyword_prediction is not None:
        if ml_prediction == 2 and keyword_prediction in (0, 1) and confidence < 0.85:
            return keyword_prediction
        if len(text.split()) <= 4 and confidence < 0.60:
            return keyword_prediction
        if confidence < 0.70 and keyword_prediction != ml_prediction:
            return keyword_prediction

    return ml_prediction


def build_stats_sections(language):
    dataset_stats = load_dataset_stats(language)
    text = LANGUAGE_TEXT[language]
    sections = [
        {
            "key": "sq",
            "title": dataset_stats["sq"]["title"],
            "total": dataset_stats["sq"]["total"],
            "items": dataset_stats["sq"]["items"],
            "note": None,
        },
        {
            "key": "en",
            "title": dataset_stats["en"]["title"],
            "total": dataset_stats["en"]["total"],
            "items": dataset_stats["en"]["items"],
            "note": text["stats_note_en"],
        },
    ]
    return sections

@app.route("/", methods=["GET", "POST"])
def home():
    language = request.values.get("language") or request.args.get("lang") or "sq"
    if language not in LANGUAGE_TEXT:
        language = "sq"

    text = LANGUAGE_TEXT[language]
    result = None
    result_key = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("comment", "")

        if user_text.strip() != "":
            prediction = predict_sentiment(user_text)
            result_key = prediction if prediction in text["labels"] else 2
            result = text["labels"].get(prediction, text["labels"][2])

    return render_template(
        "index.html",
        language=language,
        texts=text,
        translations=LANGUAGE_TEXT,
        result=result,
        result_key=result_key,
        user_text=user_text,
        stats_sections=build_stats_sections(language),
    )

if __name__ == "__main__":
    app.run(debug=True)
