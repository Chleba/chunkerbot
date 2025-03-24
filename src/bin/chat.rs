use reqwest::Url;
use std::sync::Arc;
use std::{env, io::Write};

use futures_util::StreamExt;
use langchain_rust::{
    chain::{Chain, ConversationalRetrieverChainBuilder},
    document_loaders::{pdf_extract_loader::PdfExtractLoader, Loader},
    embedding::{Embedder, OllamaEmbedder},
    fmt_message, fmt_template,
    llm::client::{GenerationOptions, Ollama, OllamaClient},
    memory::SimpleMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{Document, Message},
    template_jinja2,
    vectorstore::{
        qdrant::{Qdrant, StoreBuilder},
        Retriever, VecStoreOptions, VectorStore,
    },
};
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;

#[tokio::main]
async fn main() {
    env_logger::init();

    // -- llm
    let ollama = Ollama::default().with_model("aya");
    let ollama_client = Arc::new(OllamaClient::from_url(
        Url::parse("http://192.168.1.159:11434").unwrap(),
    ));
    let ollama = Ollama::new(
        ollama_client.clone(),
        "gemma3:12b",
        Some(GenerationOptions::default()),
    );

    let msg_template = template_jinja2!(
        "Odpovez na otazku pouze z tohoto textu: {{context}}
    Otazka: {{question}}",
        "context",
        "question"
    );

    let ollama_embed = OllamaEmbedder::new(
        ollama_client.clone(),
        "paraphrase-multilingual",
        Some(GenerationOptions::default()),
    );
    let db_client = Qdrant::from_url("http://localhost:6334").build().unwrap();
    let vector_store = StoreBuilder::new()
        .embedder(ollama_embed)
        .client(db_client)
        .collection_name("documents")
        .build()
        .await
        .unwrap();

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message("Jsi AI pomocnik ve firme S&W pro strucne odpovedi na dotazy z dodanych documents internich smernic. Odpovidej co nepresneji dle dodaneho kontextu.")),
        fmt_template!(HumanMessagePromptTemplate::new(msg_template))
    ];
    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(ollama)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(vector_store, 5))
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    loop {
        // Ask for user input
        println!("");
        print!("Query> ");
        std::io::stdout().flush().unwrap();
        let mut query = String::new();
        std::io::stdin().read_line(&mut query).unwrap();

        let query = query.trim(); // Trim input to avoid issues with empty queries
        if query.is_empty() {
            println!("Empty query. Exiting...");
            break;
        }

        let input_variables = prompt_args! {
            "question" => &query,
        };

        let mut stream = chain.stream(input_variables).await.unwrap();
        while let Some(result) = stream.next().await {
            match result {
                Ok(data) => data.to_stdout().unwrap(),
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
        }
    }
}
