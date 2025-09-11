import string

SPECIAL_TOKENS = {"So", "Let", "Hmm", "I", "Okay", "First", "Wait", "But", "Now", "Then",
                  "Since", "Therefore", "If", "Maybe", "To"}

def normalize_token_for_match(tok: str) -> str:
    """
    Try to make tokenizer pieces comparable to plain words:
    - strip leading whitespace
    - strip common SentencePiece/BPE markers
    - strip leading punctuation
    - keep case (list is capitalized), but also try a capitalized fallback
    """
    t = tok.lstrip()  # leading spaces
    t = t.lstrip("▁Ġ")  # common markers
    t = t.lstrip(string.punctuation)
    return t
