"""Word cloud image generation."""
import re
import io


def generate_wordcloud(text: str, width: int = 800, height: int = 400,
                       background_color: str = "white") -> bytes:
    """
    Generate a word cloud image from text.

    Returns:
        PNG image as bytes (ready for Streamlit st.image or API response).
    """
    from wordcloud import WordCloud

    # Clean text
    clean = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        collocations=False,
        max_words=200,
        min_word_length=3,
    ).generate(clean)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
