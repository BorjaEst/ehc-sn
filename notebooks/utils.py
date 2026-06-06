import base64

import requests
from IPython.display import Image


def display_mermaid(mermaid_code: str):
    """
    Encodes Mermaid syntax and fetches a static PNG from mermaid.ink.
    This guarantees flawless rendering when exporting to PDF via nbconvert.
    """
    # 1. Clean the input and encode to base64
    cleaned_code = mermaid_code.strip()
    encoded_bytes = base64.urlsafe_b64encode(cleaned_code.encode("utf-8"))
    base64_string = encoded_bytes.decode("utf-8")

    # 2. Construct the mermaid.ink endpoint URL
    ink_url = f"https://mermaid.ink/img/{base64_string}"

    try:
        # 3. Fetch the image content directly
        response = requests.get(ink_url, timeout=10)
        response.raise_for_status()

        # 4. Return an IPython Image object embedded as raw data
        return Image(data=response.content, format="png")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching diagram from mermaid.ink: {e}")
        return None
