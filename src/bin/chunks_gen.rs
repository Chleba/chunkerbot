use std::fs;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain},
    document_loaders::{pdf_extract_loader::PdfExtractLoader, Loader},
    embedding::OllamaEmbedder,
    fmt_template,
    llm::client::{GenerationOptions, Ollama, OllamaClient},
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::Document,
    template_jinja2,
    vectorstore::{
        qdrant::{Qdrant, StoreBuilder},
        VecStoreOptions, VectorStore,
    },
};
use reqwest::Url;
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;

pub const CONTEXT_CHUNK_STR: &str = "
Jsi asistent pro zpracování textu. Tvým úkolem je rozšířit daný chunk textu pomocí kontextu z celého dokumentu tak, aby byl co nejvíce srozumitelný a informativní i při samostatném použití. Doplněním kontextu zajistíš, že chunk obsahuje klíčové informace, které mu chybí, a zároveň zůstane stručný a relevantní.

Vstup:

Celý dokument:
{{document}}  

Původní chunk:
{{input}}  

Požadavky na výstup:
    Doplnění kontextu – Pokud chunk odkazuje na nejasné subjekty, události nebo pojmy, doplň je z kontextu celého dokumentu.
    Konzistence – Zachovej styl a terminologii dokumentu.
    Stručnost – Chunk nesmí být příliš dlouhý, ale měl by obsahovat všechny klíčové informace.
    Koherence – Chunk by měl dávat smysl i sám o sobě, bez nutnosti číst celý dokument.

Výstup:
Vrátíš přeformulovaný chunk s doplněným kontextem. Nepřidávej žádné zbytečné informace, které nejsou v dokumentu.
";

fn get_pdf_files(directory: &str) -> Vec<String> {
    let mut pdf_files = Vec::new();
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "pdf" {
                        if let Some(path_str) = path.to_str() {
                            pdf_files.push(path_str.to_string());
                        }
                    }
                }
            }
        }
    }
    pdf_files
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // -------------------------------------
    // -- VARIABLES
    let documents = get_pdf_files("./assets");
    println!("{:?} - documents", documents);

    let ollama_client = Arc::new(OllamaClient::from_url(
        // Url::parse("http://192.168.1.159:11434").unwrap(),
        Url::parse("http://127.0.0.1:11434").unwrap(),
    ));
    let ollama = Ollama::new(
        ollama_client.clone(),
        "gemma3:12b",
        Some(GenerationOptions::default()),
    );

    let chunk_msg_template = template_jinja2!(CONTEXT_CHUNK_STR, "document", "input");
    let prompt = message_formatter![fmt_template!(HumanMessagePromptTemplate::new(
        chunk_msg_template
    ))];
    let chain = ConversationalChainBuilder::new()
        .llm(ollama.clone())
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    for doc_path in documents {
        // -------------------------------------
        // -- documents loader text extractor
        let loader = PdfExtractLoader::from_path(doc_path).unwrap();
        let doc = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;
        log::info!("{:?}", doc);

        // -------------------------------------
        // -- spliting into a meaningful chunks
        let mut chunks_vec: Vec<Document> = vec![];
        let mut doc_text: String = "".to_string();

        let tokenizer = cl100k_base().unwrap();
        let max_tokens = 800;
        let chunk_config = ChunkConfig::new(max_tokens).with_sizer(tokenizer);
        let splitter = TextSplitter::new(chunk_config);
        for doc_entry in doc.iter() {
            doc_text += &doc_entry.page_content;
            let chunks = splitter
                .chunks(&doc_entry.page_content)
                .map(|d| Document::new(d))
                .collect::<Vec<_>>();
            chunks_vec.extend(chunks);
        }

        // -------------------------------------
        // -- rephrase document to questions with contextual wrapping
        let mut context_chunks: Vec<Document> = vec![];
        for chunk in chunks_vec.iter() {
            let input_vars = prompt_args! {
                "document" => doc_text,
                "input" => &chunk.page_content,

            };

            println!("----------------------------\n");
            println!("{:?} - chunk", chunk.page_content);

            match chain.invoke(input_vars).await {
                Ok(result) => {
                    println!("----------------------------\n");
                    println!("Result: {:?}", result);
                    context_chunks.push(Document::new(result));
                }
                Err(e) => panic!("Error invoking LLMChain: {:?}", e),
            }

            // -------------------------------------
            // -- sleep between chunks so poor GPU don't blow up
            tokio::time::sleep(Duration::from_secs(20)).await;
        }

        // -------------------------------------
        // -- embeddings & vector store
        let db_client = Qdrant::from_url("http://localhost:6334").build().unwrap();
        let ollama_embed = OllamaEmbedder::new(
            ollama_client.clone(),
            "paraphrase-multilingual",
            Some(GenerationOptions::default()),
        );
        let vector_store = StoreBuilder::new()
            .embedder(ollama_embed)
            // .recreate_collection(true)
            .client(db_client)
            .collection_name("documents")
            .build()
            .await
            .unwrap();
        vector_store
            .add_documents(&context_chunks, &VecStoreOptions::default())
            .await
            .unwrap();
    }
}
