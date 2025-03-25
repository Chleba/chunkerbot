use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use unescape::unescape;

use std::{collections::HashMap, fs, io::Write, sync::Arc, time::Duration};
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;

use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain, ConversationalRetrieverChainBuilder},
    document_loaders::{pdf_extract_loader::PdfExtractLoader, Loader},
    embedding::OllamaEmbedder,
    fmt_message, fmt_template,
    llm::client::{GenerationOptions, Ollama, OllamaClient},
    memory::SimpleMemory,
    message_formatter, output_parsers,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{Document, Message, Retriever},
    template_jinja2,
    vectorstore::{
        qdrant::{Qdrant, StoreBuilder},
        // Retriever, VecStoreOptions, VectorStore,
        VecStoreOptions,
        VectorStore,
    },
};

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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    Chat,
    Generate,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // chatting and generating model
    #[arg(short, long, default_value = "gemma3:12b")]
    model: Option<String>,
    // embedding model
    #[arg(short, long, default_value = "paraphrase-multilingual")]
    embed: Option<String>,
    // qdrant db url
    #[arg(long, default_value = "http://localhost:6334")]
    db: Option<String>,
    #[arg(short, long)]
    document: Option<String>,
    #[arg(short, long, default_value = "http://localhost:11434")]
    ollama: Option<String>,
    #[arg(value_enum)]
    mode: Mode,
}

async fn chat(ollama_url: String, model: String, embed: String, db_url: String) {
    // -- llm
    let ollama_client = Arc::new(OllamaClient::from_url(Url::parse(&ollama_url).unwrap()));
    let ollama = Ollama::new(
        ollama_client.clone(),
        &model,
        Some(GenerationOptions::default()),
    );

    let msg_template = template_jinja2!(
        "
    Jsi pokročilý AI asistent, který odpovídá na otázky na základě poskytnutého kontextu.  
    Tvoje úloha je analyzovat poskytnuté informace a vybrat **pouze ty nejrelevantnější** pro odpověď.  

    📌 **Otázka uživatele:**  
    {{question}}

    📌 **Poskytnuté informace (může obsahovat irelevantní části):**  
    {{context}}

    📌 **Instrukce pro odpověď:**  
    1. **Používej historii konverzace k udržení kontextu.** Pokud otázka odkazuje na předchozí část dialogu, zohledni ji.  
    2. **Pečlivě vyhodnoť, které části poskytnutého textu jsou relevantní.** Nepoužívej irelevantní informace.  
    3. **Odpověz podrobně a strukturovaně.** Pokud je to vhodné, použij odstavce, seznamy nebo příklady.  
    4. **Zahrň související informace, které mohou být užitečné pro odpověď.**  
    5. **Nevyužívej žádné jiné znalosti mimo poskytnutý kontext a historii konverzace.**  
    6. **Pokud v poskytnutých informacích odpověď chybí, přiznej to, ale nabídni užitečné doplňující informace, pokud to dává smysl.**  

    **Tvoje odpověď:**",
    "context",
    "question"
    );

    // let msg_template = template_jinja2!(
    //     "Odpoved na otazku pouze z tohoto textu: {{context}}.
    // Otazka: {{question}}",
    //     "context",
    //     "question"
    // );

    let ollama_embed = OllamaEmbedder::new(
        ollama_client.clone(),
        &embed,
        Some(GenerationOptions::default()),
    );
    let db_client = Qdrant::from_url(&db_url).build().unwrap();
    let vector_store = StoreBuilder::new()
        .recreate_collection(false)
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
    let retviever = langchain_rust::vectorstore::Retriever::new(vector_store, 5)
        .with_options(VecStoreOptions::new().with_score_threshold(0.5));
    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(ollama)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(retviever)
        .return_source_documents(true)
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    loop {
        // Ask for user input
        println!("\n");
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

        if let result = chain.execute(input_variables).await {
            match result {
                Ok(data) => {
                    let output = data["output"].as_str().unwrap();
                    let out_formatted = unescape(output).unwrap();

                    let mut used_docs: Vec<String> = data["source_documents"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| {
                            // format!("{} (s:{})", d["metadata"]["path"], d["score"])
                            format!("{}", d["metadata"]["path"])
                            // let mut d_str = d["metadata"]["path"].as_str().unwrap().to_string();
                        })
                        .collect();
                    used_docs.sort();
                    used_docs.dedup();

                    println!("{}", out_formatted);
                    println!("-------\ndocuments:[{}]", used_docs.join(", "));
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
        };

        // let mut stream = chain.stream(input_variables).await.unwrap();
        // while let Some(result) = stream.next().await {
        //     match result {
        //         Ok(data) => data.to_stdout().unwrap(),
        //         Err(e) => {
        //             println!("Error: {:?}", e);
        //         }
        //     }
        // }
    }
}

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

async fn generate(
    document: String,
    ollama_url: String,
    model: String,
    embed: String,
    db_url: String,
) {
    // -------------------------------------
    // -- VARIABLES
    // let documents = get_pdf_files("./assets");
    // println!("{:?} - documents", documents);

    let documents = vec![document];

    let ollama_client = Arc::new(OllamaClient::from_url(Url::parse(&ollama_url).unwrap()));
    let ollama = Ollama::new(
        ollama_client.clone(),
        &model,
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
        let loader = PdfExtractLoader::from_path(doc_path.clone()).unwrap();
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
        let max_tokens = 512;
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

            println!("----------------------------");
            println!("CHUNK:");
            println!("{:?}", chunk.page_content);
            println!("---\n");

            match chain.invoke(input_vars).await {
                Ok(result) => {
                    println!("RESULT:");
                    println!("{:?}", result);
                    let mut h = HashMap::new();
                    h.insert("path".to_string(), Value::String(doc_path.clone()));
                    let d = Document::new(result).with_metadata(h);
                    context_chunks.push(d);
                }
                Err(e) => panic!("Error invoking LLMChain: {:?}", e),
            }

            // -------------------------------------
            // -- sleep between chunks so poor GPU don't blow up
            // tokio::time::sleep(Duration::from_secs(20)).await;
        }

        // -------------------------------------
        // -- embeddings & vector store
        let db_client = Qdrant::from_url(&db_url).build().unwrap();
        let ollama_embed = OllamaEmbedder::new(
            ollama_client.clone(),
            &embed,
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

#[tokio::main]
async fn main() {
    env_logger::init();

    let cli = Cli::parse();
    match cli.mode {
        Mode::Chat => {
            chat(
                cli.ollama.unwrap(),
                cli.model.unwrap(),
                cli.embed.unwrap(),
                cli.db.unwrap(),
            )
            .await;
        }
        Mode::Generate => {
            if cli.document.is_none() {
                println!("Missing document for generating chunks. \nAdd --document [path_to_document] into aruments.");
                return;
            }
            generate(
                cli.document.unwrap(),
                cli.ollama.unwrap(),
                cli.model.unwrap(),
                cli.embed.unwrap(),
                cli.db.unwrap(),
            )
            .await;
        }
    }
}
