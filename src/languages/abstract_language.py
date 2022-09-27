

class Language:

    symbol = ""
    alphabet = ""
    valid_chars = ".,:;'\"()[]{}?!@#$%&- \t"

    @classmethod
    def pre_translation_callback(cls, text: str) -> str:
        return text

    @classmethod
    def post_translation_callback(cls, text: str) -> str:
        return text

    @classmethod
    def is_lang(cls, text: str) -> bool:
        count = 0
        language_symbols = cls.valid_chars + cls.alphabet
        max_count = min(len(text), 20)
        for c in text[:max_count]:
            count += c in language_symbols

        return count / max_count > 0.8



