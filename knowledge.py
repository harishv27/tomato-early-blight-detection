"""Knowledge base: a small SQLite database with advisory text for each diagnosis."""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.db")

# List fields are stored one item per line.
RECORDS = [
    {
        "disease_key": "early_blight",
        "name": "Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "description": (
            "Early blight is a common fungal disease of tomato. It usually starts on the older, "
            "lower leaves and moves upward, and spreads faster in warm, humid weather and on wet foliage."
        ),
        "symptoms": "\n".join([
            "Small dark brown to black spots on the older, lower leaves.",
            "Spots enlarge and show concentric rings, giving a target-board pattern.",
            "Yellowing of the leaf tissue around the spots.",
            "Heavily infected leaves dry up and fall, exposing the fruit to sun scald.",
        ]),
        "organic": "\n".join([
            "Remove and destroy the infected lower leaves; do not compost them.",
            "Spray neem oil at about 5 ml per litre of water at 7 to 10 day intervals.",
            "Apply bio-control agents such as Trichoderma viride or Bacillus subtilis.",
            "Use an approved copper-based spray such as Bordeaux mixture (1%).",
        ]),
        "chemical": "\n".join([
            "Mancozeb 75% WP at about 2 to 2.5 g per litre of water.",
            "Chlorothalonil 75% WP at about 2 g per litre of water.",
            "Copper oxychloride 50% WP at about 3 g per litre of water.",
            "Azoxystrobin 23% SC at about 1 ml per litre of water for severe infection.",
            "Rotate fungicide groups and follow the product label and waiting period.",
        ]),
        "prevention": "\n".join([
            "Follow a 2 to 3 year crop rotation with non-solanaceous crops.",
            "Use certified disease-free seed and tolerant varieties.",
            "Stake the plants and keep enough spacing for air movement.",
            "Use drip irrigation and avoid wetting the leaves.",
            "Mulch the soil to stop spores splashing onto the lower leaves.",
            "Remove crop debris and weeds after harvest.",
        ]),
    },
    {
        "disease_key": "healthy",
        "name": "Healthy",
        "pathogen": "None",
        "description": "No early blight lesions were found on this leaf.",
        "symptoms": "",
        "organic": "",
        "chemical": "",
        "prevention": "\n".join([
            "Keep inspecting the lower leaves every week, especially in warm, humid weather.",
            "Water at the base of the plant and avoid wetting the leaves.",
            "Maintain spacing, staking and mulching for good air movement.",
            "Apply balanced nutrition; stressed plants are more prone to early blight.",
        ]),
    },
]

LIST_FIELDS = ("symptoms", "organic", "chemical", "prevention")


def init_db():
    """Create the database and (re)load the advisory records."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS disease (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   disease_key TEXT UNIQUE NOT NULL,
                   name TEXT NOT NULL,
                   pathogen TEXT,
                   description TEXT,
                   symptoms TEXT,
                   organic TEXT,
                   chemical TEXT,
                   prevention TEXT
               )"""
        )
        con.executemany(
            """INSERT INTO disease (disease_key, name, pathogen, description, symptoms, organic, chemical, prevention)
               VALUES (:disease_key, :name, :pathogen, :description, :symptoms, :organic, :chemical, :prevention)
               ON CONFLICT(disease_key) DO UPDATE SET
                   name=excluded.name, pathogen=excluded.pathogen, description=excluded.description,
                   symptoms=excluded.symptoms, organic=excluded.organic, chemical=excluded.chemical,
                   prevention=excluded.prevention""",
            RECORDS,
        )


def get_advisory(disease_key):
    """Return the advisory record for a diagnosis as a dict (list fields split into lists)."""
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM disease WHERE disease_key = ?", (disease_key,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    for field in LIST_FIELDS:
        record[field] = [line for line in (record[field] or "").split("\n") if line]
    return record
