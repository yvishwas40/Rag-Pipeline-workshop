"""
RAG: retrieve passages from your indexed PDF, then ask Gemini to answer.
"""

import config
import rag_pipeline


def main():
    print("Chat with your document. Type quit to stop.\n")

    index = rag_pipeline.open_index()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "uit"):
            print("Bye.")
            break

        matches = rag_pipeline.search(question, index, config.TOP_K)
        chunks = rag_pipeline.get_chunk_texts(matches)
        prompt = rag_pipeline.build_prompt(question, chunks)
        answer = rag_pipeline.ask_gemini(prompt)

        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
