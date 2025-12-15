import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

function fatalError(message: string, error?: unknown) {
  console.error(message, error);

  const el =
    document.getElementById("error-label") ??
    document.getElementById("generate-label");

  if (el) {
    el.innerText =
      "Error occurred:\n" +
      message +
      (error instanceof Error ? `\n\n${error.message}` : "");
  }

  const sendBtn = document.getElementById("send-label") as HTMLButtonElement | null;
  if (sendBtn) sendBtn.disabled = true;

  throw error ?? new Error(message);
}


async function main() {
  try {
    const promptEl = document.getElementById("prompt-label") as HTMLTextAreaElement;
    const outputEl = document.getElementById("generate-label")!;
    const sendBtn = document.getElementById("send-label")!;

    sendBtn.disabled = true;
    
    // Initialize WebLLM engine
    setLabel("init-label", "Initializing WebLLM...");
    const initProgressCallback = (report: webllm.InitProgressReport) => {
        setLabel("init-label", report.text);
    };
    const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
    const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
        selectedModel,
        {
        initProgressCallback: initProgressCallback,
        logLevel: "INFO", // specify the log level
        },
        {
        context_window_size: 2048,
        // sliding_window_size: 1024,
        // attention_sink_size: 4,
        },
    );

    // Setup prompt and run chat competition
    const messages: webllm.ChatCompletionMessageParam[] = [];
    sendBtn.disabled = false;
    sendBtn.onclick = async () => {
        const userPrompt = promptEl.value.trim();
        if (!userPrompt) return;

        messages.push({ role: "user", content: userPrompt });

        sendBtn.disabled = true;

        const stream = await engine.chat.completions.create({
        messages,
        stream: true,
        });

        let assistantText = "";
        for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) {
            assistantText += delta;
            outputEl.innerText = assistantText;
        }
        }
        promptEl.value = "";
        sendBtn.disabled = false;
    };
  } catch (error) {
    fatalError("Failed to initialize WebLLM", error);
  }
}

main();