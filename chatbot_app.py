import pandas as pd
import streamlit as st
import tempfile
import os
import numpy as np
from difflib import SequenceMatcher
try:
    from langchain_openai import ChatOpenAI  # Updated import
    from langchain.memory import ConversationBufferMemory
    from langchain_community.document_loaders import UnstructuredExcelLoader, CSVLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings  # Updated import
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
except ImportError as e:
    st.error(f"Missing LangChain packages. Please install: {str(e)}")
    st.info("Run: pip install langchain langchain-community langchain-experimental langchain-openai")
    st.stop()

class PlayerDataProcessor:
    def __init__(self):
        self.similarity_threshold = 0.8
    
    def create_full_name(self, df, first_col, last_col):
        """Create full name column from first and last name"""
        return df[first_col].str.strip() + " " + df[last_col].str.strip()
    
    def calculate_name_similarity(self, name1, name2):
        """Calculate similarity between two names"""
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def process_excel_files(self, player_file_path, stats_file_path, 
                          player_first_col, player_last_col, 
                          stats_first_col, stats_last_col):
        """Process both Excel files and create merged dataset with player_id"""
        
        # Load the Excel files
        try:
            if player_file_path.endswith('.csv'):
                players_df = pd.read_csv(player_file_path)
            else:
                players_df = pd.read_excel(player_file_path)
                
            if stats_file_path.endswith('.csv'):
                stats_df = pd.read_csv(stats_file_path)
            else:
                stats_df = pd.read_excel(stats_file_path)
        except Exception as e:
            raise Exception(f"Error loading files: {str(e)}")
        
        # Create full names
        players_df['full_name'] = self.create_full_name(players_df, player_first_col, player_last_col)
        stats_df['full_name'] = self.create_full_name(stats_df, stats_first_col, stats_last_col)
        
        # Get unique names from both datasets and generate IDs
        all_names = pd.concat([players_df['full_name'], stats_df['full_name']]).unique()
        name_to_id = {name: idx + 1 for idx, name in enumerate(all_names)}
        
        # Add player_id to both dataframes
        players_df['player_id'] = players_df['full_name'].map(name_to_id)
        stats_df['player_id'] = stats_df['full_name'].map(name_to_id)
        
        # Merge the dataframes on player_id
        merged_df = pd.merge(players_df, stats_df, on='player_id', how='outer', suffixes=('', '_stats'))
        
        # Clean up duplicate columns
        if 'full_name_stats' in merged_df.columns:
            merged_df = merged_df.drop('full_name_stats', axis=1)
        
        return players_df, stats_df, merged_df, name_to_id

class MultiExcelChatbot:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.processor = PlayerDataProcessor()

    def load_excel_files(self, file_paths):
        loaders = []
        for file_path in file_paths:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                loader = UnstructuredExcelLoader(file_path, mode="elements")
                loaders.append(loader)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
                loaders.append(loader)
        merged_loader = MergedDataLoader(loaders=loaders)
        documents = merged_loader.load()
        return documents

    def create_vectorstore(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore

    def setup_qa_chain(self, vectorstore):
        retriever = vectorstore.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )
        return qa_chain

    def create_pandas_agent(self, df):
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            return_intermediate_steps=False
        )
        return agent

def main():
    st.set_page_config(page_title="Player Statistics Chatbot", page_icon="⚽", layout="wide")
    st.title("⚽ Player Statistics Chatbot with Name Matching")
    st.markdown("Upload your Excel files with player data (no player_id required!) and ask questions!")

    # Sidebar for API key and configuration
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to activate chat features"
        )
        
        st.subheader("Name Matching Settings")
        similarity_threshold = st.slider(
            "Name Similarity Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8, 
            step=0.1,
            help="Higher values require more exact name matches"
        )

    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        return

    st.header("📁 Upload Excel Files")
    
    # File uploads with column selection
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Player Information File")
        player_file = st.file_uploader(
            "Upload Player Names Excel/CSV",
            type=['xlsx', 'xls', 'csv'],
            key="player_file"
        )
        
        if player_file:
            try:
                if player_file.name.endswith('.csv'):
                    temp_df = pd.read_csv(player_file)
                else:
                    temp_df = pd.read_excel(player_file)
                
                player_first_col = st.selectbox(
                    "Select First Name Column:",
                    temp_df.columns.tolist(),
                    key="player_first"
                )
                player_last_col = st.selectbox(
                    "Select Last Name Column:",
                    temp_df.columns.tolist(),
                    key="player_last"
                )
                player_file.seek(0)
                
            except Exception as e:
                st.error(f"Error reading player file: {str(e)}")
                player_first_col = None
                player_last_col = None
        else:
            player_first_col = None
            player_last_col = None

    with col2:
        st.subheader("Season Statistics File")
        stats_file = st.file_uploader(
            "Upload Season Statistics Excel/CSV",
            type=['xlsx', 'xls', 'csv'],
            key="stats_file"
        )
        
        if stats_file:
            try:
                if stats_file.name.endswith('.csv'):
                    temp_df = pd.read_csv(stats_file)
                else:
                    temp_df = pd.read_excel(stats_file)
                
                stats_first_col = st.selectbox(
                    "Select First Name Column:",
                    temp_df.columns.tolist(),
                    key="stats_first"
                )
                stats_last_col = st.selectbox(
                    "Select Last Name Column:",
                    temp_df.columns.tolist(),
                    key="stats_last"
                )
                stats_file.seek(0)
                
            except Exception as e:
                st.error(f"Error reading stats file: {str(e)}")
                stats_first_col = None
                stats_last_col = None
        else:
            stats_first_col = None
            stats_last_col = None

    # Process files when ready
    if (player_file and stats_file and 
        player_first_col and player_last_col and 
        stats_first_col and stats_last_col):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            player_path = os.path.join(temp_dir, player_file.name)
            stats_path = os.path.join(temp_dir, stats_file.name)

            with open(player_path, "wb") as f:
                f.write(player_file.read())
            with open(stats_path, "wb") as f:
                f.write(stats_file.read())

            # Initialize chatbot and processor
            chatbot = MultiExcelChatbot(openai_api_key)
            chatbot.processor.similarity_threshold = similarity_threshold

            try:
                with st.spinner("Processing files and matching names..."):
                    players_df, stats_df, merged_df, name_to_id = chatbot.processor.process_excel_files(
                        player_path, stats_path,
                        player_first_col, player_last_col,
                        stats_first_col, stats_last_col
                    )

                # Display processing results
                st.header("📊 Data Processing Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Players Found", len(name_to_id))
                with col2:
                    st.metric("Player Records", len(players_df))
                with col3:
                    st.metric("Statistics Records", len(stats_df))

                # Show name-to-ID mapping
                st.subheader("🆔 Generated Player IDs")
                id_mapping_df = pd.DataFrame(list(name_to_id.items()), columns=['Player Name', 'Player ID'])
                st.dataframe(id_mapping_df, use_container_width=True)

                # Display merged data preview
                st.subheader("📋 Merged Data Preview")
                st.dataframe(merged_df.head(10), use_container_width=True)

                # Create chatbot components
                pandas_agent = chatbot.create_pandas_agent(merged_df)
                
                # Chat Interface
                st.header("💬 Chat with Your Player Data")
                
                method = st.radio(
                    "Select Query Method:",
                    ["Pandas Agent (Structured Analysis)", "Vector Search (Semantic Search)"]
                )

                user_question = st.text_input(
                    "Ask a question about your player data:",
                    placeholder="e.g., Which player scored the most goals in 2024?"
                )

                if user_question:
                    with st.spinner("Processing your question..."):
                        try:
                            if method == "Pandas Agent (Structured Analysis)":
                                response = pandas_agent.run(user_question)
                                st.success("**Answer:**")
                                st.write(response)
                            else:  # Vector Search
                                # Save processed files for vector search
                                processed_player_path = os.path.join(temp_dir, "processed_players.csv")
                                processed_stats_path = os.path.join(temp_dir, "processed_stats.csv")
                                
                                players_df.to_csv(processed_player_path, index=False)
                                stats_df.to_csv(processed_stats_path, index=False)
                                
                                documents = chatbot.load_excel_files([processed_player_path, processed_stats_path])
                                vectorstore = chatbot.create_vectorstore(documents)
                                qa_chain = chatbot.setup_qa_chain(vectorstore)
                                
                                result = qa_chain({"question": user_question})
                                st.success("**Answer:**")
                                st.write(result['answer'])
                                
                                if result.get('source_documents'):
                                    st.info("**Sources:**")
                                    for doc in result['source_documents'][:2]:
                                        st.write(f"- {doc.page_content[:200]}...")
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")

                # Example Questions
                st.header("💡 Example Questions")
                examples = [
                    "Who scored the most goals in 2024?",
                    "What is the average age of forwards?",
                    "Which player has the most assists across all seasons?",
                    "Show me John Smith's complete statistics",
                    "Compare performance between teams",
                    "Who has the most yellow cards?",
                    "List all goalkeepers and their statistics"
                ]

                cols = st.columns(2)
                for i, example in enumerate(examples):
                    col = cols[i % 2]
                    if col.button(example, key=f"example_{i}"):
                        st.text_input(
                            "Auto-filled question:", 
                            value=example, 
                            key=f"auto_question_{i}"
                        )

                # Download processed data
                st.header("📥 Download Processed Data")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="Download Players with IDs",
                        data=players_df.to_csv(index=False),
                        file_name="players_with_ids.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="Download Stats with IDs",
                        data=stats_df.to_csv(index=False),
                        file_name="stats_with_ids.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    st.download_button(
                        label="Download Merged Data",
                        data=merged_df.to_csv(index=False),
                        file_name="merged_player_data.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

    else:
        st.info("Please upload both Excel files and select the name columns to continue.")

if __name__ == "__main__":
    main()
