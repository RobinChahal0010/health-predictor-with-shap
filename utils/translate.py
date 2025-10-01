import json

with open("translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

def t(lang, key):
    return translations.get(lang, translations["en"]).get(key, key)
