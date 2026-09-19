"""
Personal Python Question-Answering Chatbot — Streamlit app
------------------------------------------------------------
Same knowledge base and matching logic as the CLI version, wrapped in a
Streamlit chat UI so it can be deployed as a shareable web app.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import difflib
import re

import streamlit as st

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE
#    Add more "question": "answer" pairs here — no other code changes needed.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "what is python": "Python is a high-level, interpreted programming language known for its readable syntax and large standard library.",
    "what is a variable": "A variable is a name that refers to a value stored in memory. Example: `x = 5`.",
    "what are python data types": "Common built-in types are int, float, str, bool, list, tuple, dict, and set.",
    "what is a list": "A list is an ordered, mutable collection of items. Example: `my_list = [1, 2, 3]`.",
    "what is a tuple": "A tuple is an ordered, immutable collection of items. Example: `my_tuple = (1, 2, 3)`.",
    "what is a dictionary": "A dictionary stores key-value pairs. Example: `d = {'name': 'Alice', 'age': 30}`.",
    "what is a set": "A set is an unordered collection of unique items. Example: `s = {1, 2, 3}`.",
    "what is a list comprehension": "A list comprehension is a concise way to build a list. Example: `squares = [x**2 for x in range(10)]`.",
    "what is a function": "A function is a reusable block of code defined with `def`. Example: `def greet(name): return f'Hello {name}'`.",
    "what is a lambda function": "A lambda is a small anonymous function defined with the `lambda` keyword. Example: `square = lambda x: x**2`.",
    "what are args and kwargs": "`*args` collects extra positional arguments into a tuple; `**kwargs` collects extra keyword arguments into a dict.",
    "what is a decorator": "A decorator is a function that wraps another function to extend or modify its behavior, applied with `@decorator_name` syntax.",
    "what is a generator": "A generator is a function that uses `yield` to produce a sequence of values lazily, one at a time, instead of returning them all at once.",
    "what is the difference between a list and a tuple": "Lists are mutable (can be changed after creation); tuples are immutable (cannot be changed after creation).",
    "what is oop": "Object-Oriented Programming (OOP) organizes code into classes and objects. Python supports encapsulation, abstraction, inheritance, and polymorphism.",
    "what is a class": "A class is a blueprint for creating objects. Example: `class Dog: def __init__(self, name): self.name = name`.",
    "what is inheritance": "Inheritance lets a class (child) reuse and extend the behavior of another class (parent). Example: `class Puppy(Dog): pass`.",
    "what is self in python": "`self` refers to the current instance of a class and is the first parameter of instance methods, used to access its attributes.",
    "what is polymorphism": "Polymorphism means different classes can define methods with the same name, and the correct one is called based on the object's type.",
    "what is encapsulation": "Encapsulation bundles data and methods that operate on it within a class, restricting direct access using naming conventions like `_var` or `__var`.",
    "what is an exception": "An exception is an error detected during execution. Python handles them with try/except blocks.",
    "how do i handle exceptions": "Use a try/except block: `try: risky_code() except SomeError as e: handle(e)`.",
    "what is the difference between except and finally": "`except` handles a raised exception; `finally` always runs afterward, whether an exception occurred or not.",
    "what is a module": "A module is a `.py` file containing Python code (functions, classes, variables) that can be imported with `import module_name`.",
    "what is a package": "A package is a directory of modules containing an `__init__.py` file, used to organize related modules together.",
    "what is pip": "pip is Python's package manager, used to install third-party libraries. Example: `pip install requests`.",
    "what is a virtual environment": "A virtual environment is an isolated Python environment with its own installed packages, created with `python -m venv env_name`.",
    "what is the difference between is and ==": "`==` compares values for equality; `is` compares object identity (whether two references point to the same object in memory).",
    "what is slicing": "Slicing extracts a portion of a sequence using `[start:stop:step]`. Example: `my_list[1:4]` returns items at index 1, 2, 3.",
    "what is the global interpreter lock": "The GIL (Global Interpreter Lock) allows only one thread to execute Python bytecode at a time in CPython, limiting true multithreaded CPU parallelism.",
    "what is pep 8": "PEP 8 is Python's official style guide, covering naming conventions, indentation, line length, and other formatting best practices.",
    "what is the difference between deep copy and shallow copy": "A shallow copy copies the outer object but shares references to nested objects; a deep copy recursively copies everything, so nested objects are independent.",
    "what is a docstring": "A docstring is a string literal placed right after a function, class, or module definition to document what it does, accessed via `.__doc__`.",
    "what does if __name__ == '__main__' mean": "This checks whether a script is being run directly (not imported), so code under it only runs when the file is executed as the main program.",
}

STOPWORDS = {
    "what", "is", "are", "a", "an", "the", "do", "i", "does", "in", "of",
    "and", "to", "you", "explain", "tell", "me", "about", "how", "can",
    "between", "vs", "or", "for", "on", "with",
}

FALLBACK_ANSWER = (
    "I'm not sure about that one yet — my knowledge base doesn't cover it. "
    "Try a topic button in the sidebar, or ask about Python basics, "
    "data types, functions, OOP, exceptions, or modules."
)


def normalize(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for matching."""
    text = text.lower().strip()
    text = re.sub(r"[?!.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def content_words(text: str) -> set:
    """Words that carry real meaning, ignoring question-phrasing stopwords."""
    return {w for w in normalize(text).split() if w not in STOPWORDS}


def keyword_match(user_question: str, questions: list) -> str | None:
    """Fallback: find the KB question with the highest content-word overlap."""
    user_words = content_words(user_question)
    if not user_words:
        return None

    best_question, best_score = None, 0.0
    for q in questions:
        q_words = content_words(q)
        if not q_words:
            continue
        overlap = user_words & q_words
        if not overlap:
            continue
        score = len(overlap) / min(len(user_words), len(q_words))
        if score > best_score:
            best_question, best_score = q, score

    return best_question if best_score >= 0.5 else None


def get_answer(user_question: str) -> str:
    normalized_input = normalize(user_question)
    questions = list(KNOWLEDGE_BASE.keys())

    # 1. Exact match
    if normalized_input in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized_input]

    # 2. Fuzzy match, cross-checked with keyword overlap
    close = difflib.get_close_matches(normalized_input, questions, n=1, cutoff=0.75)
    if close and keyword_match(user_question, [close[0]]):
        return KNOWLEDGE_BASE[close[0]]

    # 3. Keyword overlap fallback
    match = keyword_match(user_question, questions)
    if match:
        return KNOWLEDGE_BASE[match]

    return FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# 2. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Personal Python Q&A", page_icon="🐍", layout="centered")

st.title("🐍 Personal Python Q&A")
st.caption("Ask a Python question. Answers come from a small built-in knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Topics I know")
    st.caption("Tap one to ask it instantly.")
    for q in KNOWLEDGE_BASE:
        label = q[0].upper() + q[1:] + "?"
        if st.button(label, key=f"topic_{q}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": label})
            st.session_state.messages.append({"role": "assistant", "content": KNOWLEDGE_BASE[q]})

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("What is a decorator?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = get_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
