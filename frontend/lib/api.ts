export type Citation = {
  chunk_id: string;
  page: number | null;
  score: number;
  excerpt: string;
};

export type ChatResponse = {
  answer: string;
  grounded: boolean;
  confidence: number;
  answer_mode: "grounded" | "insufficient_context";
  citations: Citation[];
  retrieved_chunks: number;
  system_notes: string[];
};

export type SystemStatus = {
  status: "ok";
  api_name: string;
  pdf_exists: boolean;
  index_ready: boolean;
  chat_model: string;
  embedding_model: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const maybeJson = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(maybeJson?.detail ?? "The server returned an unexpected error.");
  }

  return (await response.json()) as T;
}

export async function fetchStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });

  return handleResponse<SystemStatus>(response);
}

// Keeping the original function in case you need a fallback
export async function askQuestion(query: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: 4,
      include_sources: true,
    }),
  });

  return handleResponse<ChatResponse>(response);
}

// --- NEW STREAMING FUNCTION ---
export async function askQuestionStream(
  query: string,
  onChunk: (text: string) => void
): Promise<ChatResponse> {
  // Notice we are pointing to a new /stream endpoint that we will build next
  const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: 4,
      include_sources: true,
    }),
  });

  if (!response.ok) {
    const maybeJson = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(maybeJson?.detail ?? "The server returned an unexpected error.");
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("Response body is not readable.");

  const decoder = new TextDecoder("utf-8");
  let done = false;
  let finalResponse: ChatResponse | null = null;

  while (!done) {
    const { value, done: readerDone } = await reader.read();
    done = readerDone;

    if (value) {
      // Decode the binary chunks into text
      const chunkString = decoder.decode(value, { stream: true });
      
      // SSE sends data formatted as "data: {JSON}\n\n"
      const lines = chunkString.split("\n").filter((line) => line.trim() !== "");
      
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.replace("data: ", "");
          
          if (dataStr === "[DONE]") continue;

          try {
            const data = JSON.parse(dataStr);
            
            // If it's a text token, send it to the UI immediately
            if (data.type === "token") {
              onChunk(data.text);
            } 
            // If it's the final metadata (citations, etc.), save it
            else if (data.type === "final") {
              finalResponse = data.payload;
            }
          } catch (e) {
            console.error("Error parsing stream data:", e, dataStr);
          }
        }
      }
    }
  }

  if (!finalResponse) {
    throw new Error("Stream closed before receiving the final citations.");
  }

  return finalResponse;
}

export async function rebuildIndex(): Promise<{ message: string; chunk_count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/ingest/rebuild`, {
    method: "POST",
  });

  return handleResponse<{ message: string; chunk_count: number }>(response);
}