import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from src.embeddings import model
from src.logger import logger
import chromadb
from rank_bm25 import BM25Okapi
from src.faiss_retriever import build_faiss_index, search_faiss
import json
import random
from datetime import datetime
from collections import defaultdict
import seaborn as sns

class IndustrySearchEngine:
    """Complete search engine with hybrid retrieval and visualization for industry documents."""
    
    def __init__(self, industry: str = "Technology", chroma_path: str = "./industry_chroma_db"):
        """
        Initialize the Industry Search Engine.
        
        Args:
            industry: Name of the industry domain
            chroma_path: Path for ChromaDB persistence
        """
        self.industry = industry
        self.chroma_path = chroma_path
        self.documents = []
        self.metadatas = []
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection_name = f"{industry.lower().replace(' ', '_')}_knowledge"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        logger.info(f"Initialized {industry} Search Engine")
        logger.info(f"ChromaDB path: {chroma_path}")
        logger.info(f"Collection: {self.collection_name}")

    def generate_industry_documents(self, n_documents: int = 50) -> tuple:
        """
        Generate realistic industry-specific documents.
        
        Returns:
            Tuple of (documents, metadata)
        """
        logger.info(f"Generating {n_documents} {self.industry} documents...")
        
        # Industry-specific templates and topics
        industry_data = {
            "Technology": {
                "topics": ["Cloud Computing", "AI/ML", "Cybersecurity", "DevOps", "Frontend", "Backend", 
                          "Database", "Networking", "IoT", "Blockchain", "API Design", "Microservices"],
                "templates": [
                    "The {topic} architecture utilizes {tech1} and {tech2} for optimal performance.",
                    "Recent advances in {topic} have led to {improvement} in enterprise systems.",
                    "Best practices for {topic} include implementing {practice1} and {practice2}.",
                    "Security considerations for {topic} require {security_measure}.",
                    "The future of {topic} will be shaped by {trend} and {trend2}.",
                    "Implementation of {topic} requires careful planning of {component1} and {component2}.",
                    "Performance metrics for {topic} show {metric_improvement} over traditional methods.",
                    "Integration challenges in {topic} include {challenge1} and {challenge2}.",
                    "Cost optimization strategies for {topic} involve {strategy1} and {strategy2}.",
                    "Scalability solutions for {topic} leverage {solution1} and {solution2}."
                ],
                "techs": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Python", "Java", "React", 
                         "Node.js", "MongoDB", "PostgreSQL", "Redis", "Kafka", "Spark", "TensorFlow"],
                "improvements": ["improved efficiency", "reduced latency", "enhanced security", 
                               "better scalability", "increased reliability"],
                "practices": ["continuous integration", "automated testing", "code review", 
                            "agile methodology", "devops culture"],
                "security_measures": ["encryption", "zero-trust architecture", "multi-factor authentication"],
                "trends": ["edge computing", "serverless", "quantum computing", "5G", "Web3"],
                "metrics": ["50% faster", "99.9% uptime", "40% cost reduction", "3x throughput"],
                "challenges": ["data privacy", "latency", "interoperability", "skill gaps"],
                "strategies": ["resource optimization", "auto-scaling", "load balancing"]
            },
            "Medical": {
                "topics": ["Diagnostic Imaging", "Electronic Health Records", "Telemedicine", 
                          "Genomics", "Drug Discovery", "Public Health", "Neuroscience"],
                "templates": [
                    "The {topic} protocol using {tech1} improves patient outcomes.",
                    "Clinical trials for {topic} show {improvement} in treatment efficacy.",
                    "Implementation of {topic} requires {requirement} for compliance.",
                    "Patient data management in {topic} uses {system} for privacy.",
                    "Advances in {topic} enable {capability} in healthcare delivery."
                ],
                "techs": ["MRI", "CT Scan", "EHR Systems", "Genomic Sequencers", "Telehealth Platforms"],
                "improvements": ["improved diagnosis", "faster recovery", "better prognosis"],
                "requirements": ["HIPAA compliance", "FDA approval", "clinical validation"],
                "systems": ["EMR", "PACS", "LIS", "RIS"],
                "capabilities": ["remote monitoring", "early detection", "personalized treatment"]
            },
            "Legal": {
                "topics": ["Contract Law", "Intellectual Property", "Corporate Law", 
                          "Employment Law", "Regulatory Compliance", "Litigation"],
                "templates": [
                    "The {topic} precedent {case_name} established {principle}.",
                    "Current regulations for {topic} include {regulation1} and {regulation2}.",
                    "Legal challenges in {topic} involve {challenge1} and {challenge2}.",
                    "Best practices for {topic} compliance include {practice1}.",
                    "Recent amendments to {topic} require {action} for enforcement."
                ]
            }
        }
        
        # Get data for industry (fallback to Technology if not found)
        data = industry_data.get(self.industry, industry_data["Technology"])
        topics = data.get("topics", ["General"])
        templates = data.get("templates", ["{topic} document {id}"])
        
        documents = []
        metadatas = []
        
        for i in range(n_documents):
            # Select random topic and template
            topic = random.choice(topics)
            template = random.choice(templates)
            
            # Generate document with placeholder substitution
            doc = template.format(
                topic=topic,
                tech1=random.choice(data.get("techs", ["technology"])),
                tech2=random.choice(data.get("techs", ["technology"])),
                improvement=random.choice(data.get("improvements", ["improvement"])),
                practice1=random.choice(data.get("practices", ["best practice"])),
                practice2=random.choice(data.get("practices", ["best practice"])),
                security_measure=random.choice(data.get("security_measures", ["security"])),
                trend=random.choice(data.get("trends", ["trend"])),
                trend2=random.choice(data.get("trends", ["trend"])),
                component1=random.choice(data.get("components", ["component"])),
                component2=random.choice(data.get("components", ["component"])),
                metric_improvement=random.choice(data.get("metrics", ["metric"])),
                challenge1=random.choice(data.get("challenges", ["challenge"])),
                challenge2=random.choice(data.get("challenges", ["challenge"])),
                strategy1=random.choice(data.get("strategies", ["strategy"])),
                strategy2=random.choice(data.get("strategies", ["strategy"])),
                solution1=random.choice(data.get("solutions", ["solution"])),
                solution2=random.choice(data.get("solutions", ["solution"])),
                case_name=f"{random.choice(['Smith', 'Jones', 'Doe'])} v. {random.choice(['Corp', 'Inc', 'LLC'])}",
                principle=random.choice(["precedent", "doctrine", "test", "standard"]),
                regulation1=f"Section {random.randint(101, 500)}",
                regulation2=f"Title {random.randint(1, 20)}",
                action=random.choice(["review", "update", "implement", "monitor"]),
                requirement=random.choice(["training", "certification", "audit"]),
                id=i
            )
            
            documents.append(doc)
            
            # Create metadata
            metadata = {
                "topic": topic,
                "industry": self.industry,
                "doc_id": f"{self.industry[:3].upper()}_{i:04d}",
                "timestamp": datetime.now().isoformat(),
                "doc_type": random.choice(["article", "report", "case", "guide", "standard"])
            }
            metadatas.append(metadata)
        
        self.documents = documents
        self.metadatas = metadatas
        
        logger.info(f"Generated {len(documents)} documents")
        return documents, metadatas

    def load_into_chromadb(self, documents: list = None, metadatas: list = None):
        """Load documents into ChromaDB."""
        if documents is None:
            documents = self.documents
        if metadatas is None:
            metadatas = self.metadatas
            
        logger.info("Loading documents into ChromaDB...")
        
        # Generate embeddings
        embeddings = model.encode(documents)
        
        # Add to ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Clear existing collection if it has documents
        try:
            existing = self.collection.get()
            if len(existing['ids']) > 0:
                self.collection.delete(ids=existing['ids'])
                logger.info("Cleared existing collection")
        except Exception as e:
            logger.info(f"Collection empty or error: {e}")
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            logger.info(f"   Added batch {i//batch_size + 1} ({batch_end - i} documents)")
        
        logger.info(f"Loaded {len(documents)} documents into ChromaDB")

    def visualize_memory(self, texts: list = None, save_path: str = "knowledge_map.png"):
        """
        Clusters and visualizes embeddings in 2D.
        
        Args:
            texts: List of texts to visualize (uses self.documents if None)
            save_path: Path to save the visualization
        """
        if texts is None:
            texts = self.documents
        
        if not texts:
            logger.warning("No documents to visualize!")
            return
        
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = model.encode(texts)
        
        # 1. Clustering (K-Means) to group similar documents
        n_clusters = min(len(texts), 8)  # Cap at 8 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 2. Dimensionality Reduction (UMAP)
        logger.info("Performing UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            n_components=2,
            random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings)
        
        # 3. Create detailed visualization
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
        ax1.set_title(f"{self.industry} Knowledge Map - Clustered View")
        ax1.set_xlabel("UMAP Dimension 1")
        ax1.set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster ID")
        
        # Add cluster labels with most common topic
        cluster_topics = defaultdict(list)
        for i, meta in enumerate(self.metadatas):
            cluster_topics[clusters[i]].append(meta.get('topic', 'Unknown'))
        
        for cluster_id in range(n_clusters):
            if cluster_id in cluster_topics:
                topics = cluster_topics[cluster_id]
                most_common = max(set(topics), key=topics.count) if topics else "Unknown"
                # Get centroid of cluster for label placement
                cluster_points = embedding_2d[clusters == cluster_id]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    ax1.annotate(
                        f"Cluster {cluster_id}\n({most_common})",
                        xy=centroid,
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
                    )
        
        # Plot 2: Topic-based view (color by topic)
        topics = [meta.get('topic', 'Unknown') for meta in self.metadatas]
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
        
        ax2.set_title(f"{self.industry} Knowledge Map - Topic View")
        ax2.set_xlabel("UMAP Dimension 1")
        ax2.set_ylabel("UMAP Dimension 2")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        logger.info(f"Knowledge map saved to {save_path}")
        
        # Analyze clustering quality
        self._analyze_clustering(embeddings, clusters, topics)
        
        return embedding_2d, clusters

    def _analyze_clustering(self, embeddings, clusters, topics):
        """Analyze the quality of clustering."""
        logger.info("\nCLUSTERING ANALYSIS:")
        logger.info("="*50)
        
        # Count documents per cluster
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            count = np.sum(clusters == cluster_id)
            cluster_topics = [topics[i] for i in range(len(topics)) if clusters[i] == cluster_id]
            unique_cluster_topics = set(cluster_topics)
            logger.info(f"Cluster {cluster_id}: {count} documents, Topics: {list(unique_cluster_topics)[:3]}")
        
        # Check if similar topics are grouped together
        topic_groups = defaultdict(list)
        for i, topic in enumerate(topics):
            topic_groups[topic].append(clusters[i])
        
        logger.info("\n📌 Topic Cluster Distribution:")
        for topic, cluster_list in topic_groups.items():
            unique_clusters_in_topic = set(cluster_list)
            if len(unique_clusters_in_topic) == 1:
                logger.info(f"{topic}: All in cluster {unique_clusters_in_topic}")
            else:
                logger.info(f"{topic}: Spread across clusters {unique_clusters_in_topic}")

    def hybrid_search(self, query: str, k: int = 3, use_rrf: bool = True):
        """
        Perform hybrid search on the industry documents.
        
        Args:
            query: Search query
            k: Number of results
            use_rrf: Whether to use RRF for fusion
            
        Returns:
            List of relevant documents
        """
        if not self.documents:
            logger.warning("No documents loaded! Run generate_industry_documents() first.")
            return []
        
        logger.info(f"Searching for: '{query}'")
        
        # 1. ChromaDB (Vector) Search
        query_embedding = model.encode([query]).tolist()
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k*2
        )
        
        vector_docs = chroma_results['documents'][0] if chroma_results['documents'] else []
        
        # 2. BM25 (Keyword) Search
        tokenized_corpus = [doc.split(" ") for doc in self.documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
        keyword_docs = [self.documents[idx] for idx in top_indices if bm25_scores[idx] > 0]
        
        # 3. Combine results
        if use_rrf:
            # Simple RRF implementation
            vector_ranked = {doc: rank for rank, doc in enumerate(vector_docs, 1)}
            keyword_ranked = {doc: rank for rank, doc in enumerate(keyword_docs, 1)}
            
            fusion_scores = {}
            all_docs = set(vector_docs + keyword_docs)
            
            for doc in all_docs:
                score = 0
                if doc in vector_ranked:
                    score += 1 / (60 + vector_ranked[doc])
                if doc in keyword_ranked:
                    score += 1 / (60 + keyword_ranked[doc])
                fusion_scores[doc] = score
            
            # Sort by score
            results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            results = [doc for doc, _ in results]
        else:
            # Simple union
            results = list(set(vector_docs + keyword_docs))[:k]
        
        logger.info(f"Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            logger.info(f"   {i}. {doc[:100]}...")
        
        return results

def run_industry_search_demo():
    """Run complete demonstration of the Industry Search Engine."""
    
    # Initialize engine for Technology industry
    engine = IndustrySearchEngine(industry="Technology")
    
    # Generate 50 documents
    print("\n" + "="*60)
    print("INDUSTRY SEARCH ENGINE DEMO")
    print("="*60)
    
    documents, metadatas = engine.generate_industry_documents(n_documents=50)
    
    # Load into ChromaDB
    engine.load_into_chromadb(documents, metadatas)
    
    # Test hybrid search
    print("\n" + "="*60)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*60)
    
    test_queries = [
        "cloud computing architecture best practices",
        "AI machine learning performance optimization",
        "cybersecurity data protection encryption",
        "microservices integration challenges",
        "database scalability solutions"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = engine.hybrid_search(query, k=3, use_rrf=True)
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc[:120]}...")
    
    # Generate and save visualization
    print("\n" + "="*60)
    print("GENERATING KNOWLEDGE MAP")
    print("="*60)
    
    # Get all documents from ChromaDB for visualization
    all_data = engine.collection.get()
    docs_to_visualize = all_data['documents'] if all_data['documents'] else documents
    
    # Update documents list for metadata
    engine.documents = docs_to_visualize
    engine.metadatas = metadatas[:len(docs_to_visualize)]
    
    embedding_2d, clusters = engine.visualize_memory(
        texts=docs_to_visualize,
        save_path="industry_knowledge_map.png"
    )
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("   - knowledge_map.png")
    print("   - industry_knowledge_map.png")
    print("   - ChromaDB stored in ./industry_chroma_db")

if __name__ == "__main__":
    run_industry_search_demo()