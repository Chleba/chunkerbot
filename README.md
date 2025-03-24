# chunk_contextor
LLM document spliting with ai agent that wrap chunks with document's context, store in vector DB with simple chat

## Install

`cargo build --release --bin chat_contextor`
You can find a binary in `target/release/chat_extractor`

## Qdrant

Install & run `qdrant` vector database docker
`docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant`

You need to start `gRpc` service for client to be able to connect to DB.

## Usage

`chunk_contextor --help` will tell you all
