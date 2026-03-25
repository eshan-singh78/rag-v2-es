"""
Interactive RAG bot. Run with: python bot.py
Type your question and press Enter. Type 'exit' or 'quit' to stop.
"""
from query_data import query_rag
import cache as query_cache

BANNER = """
╔══════════════════════════════════════╗
║           RAG Assistant              ║
║  Ask anything about your documents.  ║
║  Type 'exit' to quit.                ║
║  Type 'clear cache' to reset cache.  ║
╚══════════════════════════════════════╝
"""


def run():
    print(BANNER)
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Bye.")
            break

        if query.lower() == "clear cache":
            query_cache.clear_all()
            print("Cache cleared.\n")
            continue

        print()
        query_rag(query)
        print()


if __name__ == "__main__":
    run()
