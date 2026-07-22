import streamlit as st
import requests
from src.logger import logger
import os
import matplotlib.pyplot as plt
from src.embeddings import model
import chromadb
import umap
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000/research"
CHROMA_PATH = "./chroma_db"


# ==============================
# Helper Functions
# ==============================

def generate_knowledge_map(collection_name: str = "research_knowledge", save_path: str = "knowledge_map.png"):
    """
    Generate a UMAP visualization of the memory/knowledge base.
    
    Args:
        collection_name: Name of the ChromaDB collection
        save_path: Path to save the visualization
    """
    try:
        logger.info("🧠 Generating knowledge map visualization...")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=collection_name)
        
        # Get all documents
        all_data = collection.get()
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        
        if not documents:
            logger.warning("No documents found in memory!")
            return None, "No documents in memory. Add some research first!"
        
        logger.info(f"📊 Found {len(documents)} documents in memory")
        
        # Generate embeddings
        embeddings = model.encode(documents)
        
        # Clustering (K-Means)
        n_clusters = min(len(documents), 8)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Dimensionality Reduction (UMAP)
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Clustered view
        scatter1 = ax1.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=clusters,
            cmap='Spectral',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        ax1.set_title("Knowledge Map - Clustered View")
        ax1.set_xlabel("UMAP Dimension 1")
        ax1.set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster ID")
        
        # Add cluster labels
        for cluster_id in range(n_clusters):
            cluster_points = embedding_2d[clusters == cluster_id]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                ax1.annotate(
                    f"Cluster {cluster_id}",
                    xy=centroid,
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
                )
        
        # Plot 2: Topic-based view (if metadata has topics)
        if metadatas and any('topic' in meta for meta in metadatas):
            topics = [meta.get('topic', 'Unknown') for meta in metadatas]
            unique_topics = list(set(topics))
            topic_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))
            topic_to_color = {topic: color for topic, color in zip(unique_topics, topic_colors)}
            
            for topic in unique_topics:
                indices = [i for i, t in enumerate(topics) if t == topic]
                if indices:
                    points = embedding_2d[indices]
                    ax2.scatter(
                        points[:, 0],
                        points[:, 1],
                        label=topic,
                        color=topic_to_color[topic],
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5
                    )
            
            ax2.set_title("Knowledge Map - Topic View")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # If no topics, show second cluster view with different coloring
            scatter2 = ax2.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=clusters,
                cmap='viridis',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax2.set_title("Knowledge Map - Alternative View")
            plt.colorbar(scatter2, ax=ax2, label="Cluster ID")
        
        ax2.set_xlabel("UMAP Dimension 1")
        ax2.set_ylabel("UMAP Dimension 2")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate statistics
        stats = {
            'total_documents': len(documents),
            'unique_topics': len(set(topics)) if metadatas else 'N/A',
            'clusters': n_clusters,
            'documents_per_cluster': [
                int(np.sum(clusters == i)) for i in range(n_clusters)
            ]
        }
        
        logger.info(f"✅ Knowledge map saved to {save_path}")
        return save_path, stats
        
    except Exception as e:
        logger.error(f"Error generating knowledge map: {e}")
        return None, f"Error: {str(e)}"

def get_memory_stats():
    """Get statistics about the current memory."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="research_knowledge")
        all_data = collection.get()
        
        return {
            'total_documents': len(all_data['documents']),
            'has_metadata': bool(all_data['metadatas']),
            'collection_name': "research_knowledge"
        }
    except:
        return {
            'total_documents': 0,
            'has_metadata': False,
            'collection_name': "research_knowledge"
        }


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Agentic Researcher AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# Sidebar Configuration
# ==============================

with st.sidebar:

    st.title("Agent Control Panel")

    st.markdown("---")

    # Application status
    st.subheader("System Status")

    st.success("API Online")
    st.info("LLM: Groq Llama 3.3 70B")

    # Memory stats
    memory_stats = get_memory_stats()
    st.metric("Memory Documents", memory_stats['total_documents'])

    st.markdown("---")

    # Agent settings
    st.subheader("Agent Settings")

    max_agents = st.slider(
        "Number of Research Agents",
        min_value=1,
        max_value=10,
        value=3
    )

    research_depth = st.selectbox(
        "Research Depth",
        [
            "Quick Summary",
            "Standard Analysis",
            "Deep Investigation"
        ]
    )

    enable_sources = st.checkbox(
        "Include Sources",
        value=True
    )

    st.markdown("---")

    # Model settings
    st.subheader("Model Configuration")

    model = st.selectbox(
        "Select LLM Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    temperature = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.markdown("---")

    # Memory Visualization Button
    st.subheader("🧠 Memory Visualization")

    if st.button(
        "Visualize Memory",
        help="Generate and display a UMAP visualization of your knowledge base",
        use_container_width=True
    ):
        with st.spinner("Generating knowledge map..."):
            image_path, result = generate_knowledge_map()
            
            if image_path and os.path.exists(image_path):
                st.session_state['knowledge_map'] = image_path
                st.session_state['map_stats'] = result
                st.success("✅ Knowledge map generated!")
            else:
                st.error(f"❌ {result}")

    # Display stats if available
    if 'map_stats' in st.session_state and st.session_state['map_stats']:
        stats = st.session_state['map_stats']
        if isinstance(stats, dict):
            st.markdown("---")
            st.subheader("📊 Map Statistics")
            st.metric("Documents", stats.get('total_documents', 0))
            st.metric("Clusters", stats.get('clusters', 0))
            if 'documents_per_cluster' in stats:
                st.caption("Documents per cluster:")
                for i, count in enumerate(stats['documents_per_cluster']):
                    st.text(f"  Cluster {i}: {count} docs")

    st.markdown("---")

    # About section
    st.subheader("📌 About")

    st.caption(
        """
        Agentic Research Assistant

        Built with:
        - FastAPI
        - Streamlit
        - Groq LLM
        - Async Agents
        - ChromaDB Memory
        - UMAP Visualization
        """
    )


# ==============================
# Main Application
# ==============================

st.title("🚀 Agentic AI Research Assistant")

st.markdown(
    """
    Enter a topic below and your AI research agents
    will collaborate to analyze the subject.
    """
)


# ==============================
# Session State
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_map" not in st.session_state:
    st.session_state.knowledge_map = None


# ==============================
# Display Knowledge Map if Available
# ==============================

if st.session_state.knowledge_map and os.path.exists(st.session_state.knowledge_map):
    with st.expander("🧠 Knowledge Map Visualization", expanded=False):
        st.image(
            st.session_state.knowledge_map,
            caption="UMAP Projection of Research Knowledge Base",
            use_column_width=True
        )
        
        # Add download button
        with open(st.session_state.knowledge_map, "rb") as file:
            st.download_button(
                label="📥 Download Knowledge Map",
                data=file,
                file_name="knowledge_map.png",
                mime="image/png"
            )


# ==============================
# Chat History
# ==============================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# User Input
# ==============================

if prompt := st.chat_input(
    "What would you like to research?"
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # API call to FastAPI backend
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        payload = {
            "topic": prompt,
            "max_analysts": max_agents
        }

        try:
            response = requests.post(
                API_URL,
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                full_response = data["answer"]
            else:
                full_response = f"Backend Error: {response.status_code}"

        except Exception as e:
            full_response = f"Connection Error: {str(e)}"

        response_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

    logger.info(f"UI received input: {prompt}")


# ==============================
# Footer
# ==============================

st.markdown("---")
st.caption(
    """
    💡 **Tip**: Click 'Visualize Memory' in the sidebar to see a 
    UMAP projection of your research knowledge base.
    """
)