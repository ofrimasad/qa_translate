from src.languages.arabic import Arabic
from src.languages.chinese import Chinese
from src.languages.german import German
from src.languages.greek import Greek
from src.languages.hindi import Hindi
from src.languages.russian import Russian
from src.languages.thai import Thai
from src.languages.turkish import Turkish
from src.languages.vietnamese import Vietnamese
from src.languages.english import English
from src.languages.hebrew import Hebrew
from src.languages.persian import Persian
from src.languages.spanish import Spanish

__all__ = [English, Spanish, Hebrew, Persian, Arabic, Chinese, German, Greek, Hindi, Russian, Thai, Turkish, Vietnamese]

LANGUAGES = {}

for lang in __all__:
    LANGUAGES[lang.symbol] = lang
