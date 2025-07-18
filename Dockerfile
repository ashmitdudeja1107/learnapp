FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://ollama.com/install.sh | sh

# Preload a small model like llama3
RUN ollama pull llama3

EXPOSE 11434

CMD ["ollama", "serve"]
