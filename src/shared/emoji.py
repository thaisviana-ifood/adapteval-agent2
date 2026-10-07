"""Emoji extraction and sentiment classification"""

import re
from collections import Counter
from typing import Any, Dict, List

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U00002600-\U000026ff"  # misc symbols
    "\U00002b00-\U00002bff"  # misc symbols & arrows (e.g. star, snowflake)
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "]"
)

POSITIVE_EMOJIS = {
    "😀", "😁", "😂", "🤣", "😃", "😄", "😆", "😊", "😇", "🙂", "🙃", "😉",
    "😍", "🥰", "😘", "😗", "😙", "😚", "☺", "😋", "😛", "😜", "🤪", "😝",
    "🤩", "🥳", "😎", "👍", "👏", "🙌", "👌", "✌", "🤞", "🎉", "🎊", "💪",
    "❤", "💕", "💖", "💗", "💓", "💞", "💘", "💝", "✨", "🌟", "⭐", "🔥",
    "👑", "🥇",
}

NEGATIVE_EMOJIS = {
    "😠", "😡", "🤬", "😞", "😟", "😣", "😖", "😫", "😩", "🥺", "😢", "😭",
    "😤", "😥", "😓", "🙁", "☹", "😔", "😕", "🙄", "😒", "😬", "🤢", "🤮",
    "😨", "😰", "😱", "😳", "💔", "👎", "😿", "😾", "😑",
}


def extract_emojis(text: str) -> List[str]:
    """Return every emoji character found in text, in order of appearance"""
    return EMOJI_PATTERN.findall(text)


def classify_emoji_sentiment(emoji: str) -> str:
    """Classify a single emoji as 'positive', 'negative' or 'neutral'"""
    if emoji in POSITIVE_EMOJIS:
        return "positive"
    if emoji in NEGATIVE_EMOJIS:
        return "negative"
    return "neutral"


def count_emoji_frequency(texts: List[str]) -> Dict[str, int]:
    """Tally raw emoji occurrences across a list of texts"""
    counter = Counter()
    for text in texts:
        counter.update(extract_emojis(text))
    return dict(counter)


def analyze_emoji_sentiment(text: str) -> Dict[str, Any]:
    """Classify every emoji in text as positive/negative/neutral"""
    emojis = extract_emojis(text)
    sentiment_counts = Counter(classify_emoji_sentiment(e) for e in emojis)

    positive_count = sentiment_counts.get("positive", 0)
    negative_count = sentiment_counts.get("negative", 0)
    neutral_count = sentiment_counts.get("neutral", 0)
    polar_total = positive_count + negative_count

    return {
        "total_emoji_count": len(emojis),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "emoji_sentiment_score": (
            positive_count / polar_total if polar_total > 0 else 0.5
        ),
    }
