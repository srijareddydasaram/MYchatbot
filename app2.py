import streamlit as st
import chromadb
import ollama

st.set_page_config(page_title="Mini RAG Q&A")

st.title("📚 Mini RAG Q&A")
st.write("Paste a document, store it in ChromaDB, and ask questions about it.")


# ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)


# Document input
document = st.text_area(
    "📄 Paste your document here",
    height=250,
    placeholder="Paste your notes, article, syllabus, etc."
)


# Add document
if st.button("➕ Add Document"):

    if not document.strip():

        st.warning("Please enter some text.")

    else:

        # Split document into chunks
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        embeddings = []

        # Create embeddings using Ollama
        for chunk in chunks:

            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )

            embeddings.append(response["embedding"])

        # Store in ChromaDB
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings
        )

        st.success(
            f"Added {len(chunks)} chunk(s) to ChromaDB."
        )


# Question
question = st.text_input(
    "❓ Ask a question about your document"
)


# Ask AI
if st.button("🤖 Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please add a document first.")

    else:

        # Create embedding for question
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=question
        )

        question_embedding = response["embedding"]

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, collection.count())
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        # Prompt for Llama
        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say:

"I don't know based on the provided document."
"""

        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("💡 Answer")

            st.write(
                response["message"]["content"]
            )

            # Show retrieved context
            with st.expander("📖 Retrieved Context"):

                for i, chunk in enumerate(retrieved_chunks):

                    st.write(
                        f"**Chunk {i + 1}:**"
                    )

                    st.write(chunk)

        except Exception as e:

            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

            st.code(str(e))


st.caption(
    "Python + Ollama + ChromaDB + Streamlit"
)