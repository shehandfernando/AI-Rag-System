"use client";

import { FormEvent, useEffect, useState, useRef } from "react";
import {
  askQuestionStream,
  ChatResponse,
  fetchStatus,
  rebuildIndex,
  SystemStatus,
} from "../lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

const STARTER_PROMPTS = [
  "Summarize the PDF.",
  "What major topics are covered in this document?",
  "Explain the main ideas in simple words.",
];

function formatPageReferences(response: ChatResponse): string | null {
  const pages = Array.from(
    new Set(
      response.citations
        .map((citation) => citation.page)
        .filter((page): page is number => page !== null),
    ),
  ).sort((a, b) => a - b);

  if (pages.length === 0) {
    return null;
  }

  if (pages.length === 1) {
    return `Based on page ${pages[0]}`;
  }

  return `Based on pages ${pages.join(", ")}`;
}

export function ChatShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  
  // NEW: Replaced useTransition with a standard state boolean for streaming
  const [isReceiving, setIsReceiving] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;

    fetchStatus()
      .then((data) => {
        if (active) {
          setStatus(data);
        }
      })
      .catch((err: Error) => {
        if (active) {
          setError(err.message);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".pdf")) {
      setError("Please select a valid PDF file.");
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
      const response = await fetch(`${baseUrl}/api/v1/ingest/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Upload failed");
      }

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      const nextStatus = await fetchStatus();
      setStatus(nextStatus);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to upload document.";
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const submitQuestion = async (nextQuery: string) => {
    const cleaned = nextQuery.trim();
    if (!cleaned) return;

    setError(null);
    setQuery("");
    setIsReceiving(true);

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    // NEW: Instantly inject an empty assistant message bubble into the chat
    setMessages((current) => [
      ...current,
      { id: userMessageId, role: "user", content: cleaned },
      { id: assistantMessageId, role: "assistant", content: "" },
    ]);

    try {
      // NEW: Call the streaming API and append text dynamically as it arrives
      const finalResponse = await askQuestionStream(cleaned, (newText) => {
        setMessages((current) =>
          current.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: msg.content + newText }
              : msg
          )
        );
      });

      // NEW: When the stream is fully finished, attach the citations
      setMessages((current) =>
        current.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, response: finalResponse }
            : msg
        )
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
      
      // Remove the empty assistant bubble if the request failed immediately
      setMessages((current) => current.filter((msg) => msg.id !== assistantMessageId));
    } finally {
      setIsReceiving(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitQuestion(query);
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      await rebuildIndex();
      const nextStatus = await fetchStatus();
      setStatus(nextStatus);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to refresh the document.";
      setError(message);
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <main className="page-shell">
      <div className="orb orb-left" />
      <div className="orb orb-right" />

      <section className="hero-card minimal">
        <div className="hero-copy">
          <span className="eyebrow">RAG System</span>
          <h1>Ask questions about your document.</h1>
          <p>Get clear answers from the PDF in a simple chat experience.</p>
        </div>

        <div className="status-panel compact">
          <div className="status-copy">
            <strong>
              {status?.index_ready ? "Document is ready" : "Preparing your document"}
            </strong>
            <p>
              {status?.index_ready
                ? "You can start asking questions now."
                : "If you updated the PDF, refresh the document first."}
            </p>
          </div>
          
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              type="file"
              accept=".pdf"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
            <button
              className="ghost-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              type="button"
            >
              {isUploading ? "Uploading..." : "Upload PDF"}
            </button>
            <button
              className="ghost-button"
              onClick={handleRefresh}
              disabled={isRefreshing || isUploading}
            >
              {isRefreshing ? "Refreshing..." : "Refresh document"}
            </button>
          </div>
        </div>
      </section>

      <section className="chat-card simple">
        <div className="card-header simple">
          <div>
            <span className="card-label">Chat</span>
            <h2>Ask anything from the PDF.</h2>
          </div>
        </div>

        <div className="starter-row">
          {STARTER_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              className="starter-chip"
              onClick={() => submitQuestion(prompt)}
              type="button"
              disabled={isReceiving}
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="messages-panel">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>Type a question below and I will answer using the PDF.</p>
            </div>
          ) : (
            messages.map((message) => {
              const pageReference = message.response
                ? formatPageReferences(message.response)
                : null;

              return (
                <article
                  className={`message-bubble ${message.role === "assistant" ? "assistant" : "user"}`}
                  key={message.id}
                >
                  <span className="message-role">
                    {message.role === "assistant" ? "Assistant" : "You"}
                  </span>
                  
                  {/* Show a blinking cursor indicator if the message is completely empty and currently receiving */}
                  {message.content === "" && isReceiving ? (
                     <p className="streaming-cursor">Looking through the PDF...</p>
                  ) : (
                     <p>{message.content}</p>
                  )}

                  {message.response && pageReference ? (
                    <p className="page-reference">{pageReference}</p>
                  ) : null}
                </article>
              );
            })
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Ask a question"
            className="composer-input"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a question about the PDF..."
            rows={3}
            value={query}
          />
          <button
            className="primary-button"
            disabled={isReceiving || !query.trim()}
            type="submit"
          >
            {isReceiving ? "Thinking..." : "Send"}
          </button>
        </form>

        {error ? <p className="error-text">{error}</p> : null}
      </section>
    </main>
  );
}