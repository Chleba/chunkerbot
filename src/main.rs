use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use reqwest::Url;
use std::{fs, io::Write, sync::Arc, time::Duration};
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;

use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain, ConversationalRetrieverChainBuilder},
    document_loaders::{pdf_extract_loader::PdfExtractLoader, Loader},
    embedding::OllamaEmbedder,
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
        "Odpovez na otazku pouze z tohoto textu: {{context}}
    Otazka: {{question}}",
        "context",
        "question"
    );

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

            println!("----------------------------");
            println!("CHUNK:");
            println!("{:?}", chunk.page_content);
            println!("---\n");

            match chain.invoke(input_vars).await {
                Ok(result) => {
                    println!("RESULT:");
                    println!("{:?}", result);
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

    //
    // let arg_matches = Command::new("embedgen")
    //     .version("0.1")
    //     .author("Lukas Chleba Franek")
    //     .about("Chunking documents and generating context chunks that is stored into a vector DB.")
    //     .arg(Arg::new("chat").help("simple CLI chat with document context"))
    //     .arg(Arg::new("generate").help(
    //         "parse given document and generate contextual chunks that is then stored into a DB",
    //     ))
    //     .arg(
    //         Arg::new("document")
    //             .help("document for generation")
    //             .requires_if("", "generate"),
    //     )
    //     .get_matches();
    //
    // let chat_flag = arg_matches.contains_id("chat");
    //
    // println!("{:?} = chat", chat_flag);

    // if arg_match

    // let chat = arg_matches.args_present("chat");

    // // -- documents text extractor
    // let path = "./assets/Interní postup - Autoreply e-mails.pdf";
    // let path1 = "./assets/Směrnice - Sales (Stepan).pdf";
    // let path2 = "./assets/Systém vnitřních zásad_12.07.2023.pdf";
    // let loader = PdfExtractLoader::from_path(path1).unwrap();
    // let doc = loader
    //     .load()
    //     .await
    //     .unwrap()
    //     .map(|d| d.unwrap())
    //     .collect::<Vec<_>>()
    //     .await;
    // log::info!("{:?}", doc);
    //
    // // -- spliting into a meaningful chunks
    // let tokenizer = cl100k_base().unwrap();
    // let max_tokens = 750;
    // let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));
    // let chunks = splitter
    //     .chunks(&doc[0].page_content)
    //     .map(|d| Document::new(d))
    //     .collect::<Vec<_>>();
    //
    // println!("{:?}", chunks);
    // println!("chunks len - {}", &chunks.len());
    //
    // // -- embeddings & vector store
    // // let ollama_embed = OllamaEmbedder::default().with_model("nomic-embed-text");
    // let ollama_embed = OllamaEmbedder::default().with_model("paraphrase-multilingual");
    // let db_client = Qdrant::from_url("http://localhost:6334").build().unwrap();
    // let vector_store = StoreBuilder::new()
    //     .embedder(ollama_embed)
    //     .recreate_collection(true)
    //     .client(db_client)
    //     .collection_name("documents")
    //     .build()
    //     .await
    //     .unwrap();
    // vector_store
    //     // .add_documents(&doc, &VecStoreOptions::default())
    //     .add_documents(&chunks, &VecStoreOptions::default())
    //     .await
    //     .unwrap();
    //
    // // -- llm
    // let ollama = Ollama::default().with_model("aya");
    //
    // let msg_template = template_jinja2!(
    //     "Odpovez na otazku pouze z tohoto textu: {{context}}
    // Otazka: {{question}}",
    //     "context",
    //     "question"
    // );
    //
    // // let msg_template = template_jinja2!(
    // //     "Answer the question based only on the following context:
    // // {{context}}
    // // Question: {{question}}",
    // //     "context",
    // //     "question"
    // // );
    //
    // // let msg_template = template_jinja2!("
    // // Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    // //
    // // {{context}}
    // //
    // // Question:{{question}}
    // // Helpful Answer:
    // //         ",
    // //                     "context","question")
    //
    // let prompt = message_formatter![
    //     fmt_message!(Message::new_system_message("Jsi AI pomocnik ve firme S&W pro strucne odpovedi na dotazy z dodanych documents internich smernic. Odpovidej co nepresneji dle dodaneho kontextu.")),
    //     fmt_template!(HumanMessagePromptTemplate::new(msg_template))
    // ];
    // let chain = ConversationalRetrieverChainBuilder::new()
    //     .llm(ollama)
    //     .rephrase_question(true)
    //     .memory(SimpleMemory::new().into())
    //     .retriever(Retriever::new(vector_store, 5))
    //     //If you want to use the default prompt remove the .prompt()
    //     //Keep in mind if you want to change the prompt; this chain need the {{context}} variable
    //     .prompt(prompt)
    //     .build()
    //     .expect("Error building ConversationalChain");
    //
    // // Ask for user input
    // print!("Query> ");
    // std::io::stdout().flush().unwrap();
    // let mut query = String::new();
    // std::io::stdin().read_line(&mut query).unwrap();
    //
    // let input_variables = prompt_args! {
    //     "question" => &query,
    // };
    //
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
