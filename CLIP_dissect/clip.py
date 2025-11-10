"""Small wrapper module to expose the OpenAI CLIP API under CLIP_dissect.clip.
It proxies tokenize and load to the installed `clip` package.
"""
try:
    import clip as _clip
except Exception as e:
    raise

# Re-export commonly used functions/objects
def load(model_name, device='cpu'):
    return _clip.load(model_name, device=device)

# tokenization helper
def tokenize(texts):
    return _clip.tokenize(texts)

# expose module-level attributes if needed
__all__ = ['load', 'tokenize']
