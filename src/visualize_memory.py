# if i wanna run a standalone visualization of the knowledge map, I can use this script. It will generate a UMAP projection of the documents in the ChromaDB collection and display it using Streamlit. The user can also download the generated image.
import streamlit as st
from src.embeddings import model
import chromadb
import umap
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def generate_knowledge_map(chroma_path: str = "./chroma_db", collection_name: str = "research_knowledge"):
    """
    Standalone function to generate and display knowledge map.
    """
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)
        
        # Get all documents
        all_data = collection.get()
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        
        if not documents:
            st.warning("No documents found in memory!")
            return None
        
        # Generate embeddings
        embeddings = model.encode(documents)
        
        # Clustering
        n_clusters = min(len(documents), 8)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # UMAP reduction
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=clusters,
            cmap='Spectral',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_title("Knowledge Base Memory Map")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter, label="Cluster ID")
        
        # Add document labels (optional - only if few documents)
        if len(documents) <= 20:
            for i, doc in enumerate(documents):
                ax.annotate(
                    f"Doc {i}",
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"knowledge_map_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI for standalone visualization
st.set_page_config(page_title="Knowledge Map Viewer", layout="wide")

st.title("🧠 Knowledge Base Memory Map")

if st.button("Refresh Memory Map"):
    with st.spinner("Generating knowledge map..."):
        image_path = generate_knowledge_map()
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="UMAP Projection of Knowledge Base", use_column_width=True)
            
            with open(image_path, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=image_path,
                    mime="image/png"
                )
        else:
            st.error("Failed to generate knowledge map. Make sure you have documents in the database.")