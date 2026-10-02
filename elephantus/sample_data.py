"""Sample conversation for the "Load sample data" button in the UI.

The history leans heavily towards Adidas (as a real chat history would) before the
user switches to Puma — exactly the situation where similarity-only RAG retrieves
stale context and memory does not.
"""

SAMPLE_MESSAGES = [
    "I love Adidas sneakers",
    "Adidas Ultraboost are the most comfortable sneakers I've ever owned",
    "I study computer science at MIT",
    "I usually buy a new pair of Adidas sneakers every spring",
    "This week I'm building a chess engine in Rust for a class project",
    "My Adidas broke after a month",
    "I'm switching to Puma",
    "I have an exam tomorrow",
]
