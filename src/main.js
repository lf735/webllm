/**
 * GEMMA 4 STUDIO - CORE RUNTIME & INTERFACE CONTROLLER
 * Supports:
 * 1. WebGPU In-Browser Engine (@mlc-ai/web-llm)
 * 2. Localhost Engine (Ollama / LM Studio / llama.cpp)
 * 3. Gemma 4 Thinking Mode / Collapsible Reasoning Blocks
 * 4. Multimodal Vision Preview & Real-Time Performance HUD
 */

// Import WebLLM from installed npm package (supported via Vite) or fallback to ESM CDN
let webllm = null;

// Application State
const state = {
  currentEngine: 'webgpu', // 'webgpu' | 'ollama'
  selectedModel: 'gemma-2-2b-it-q4f32_1-MLC',
  thinkingModeEnabled: true,
  isGenerating: false,
  abortController: null,
  activeConversationId: null,
  conversations: [],
  attachedImage: null, // { base64: string, name: string }
  
  // LLM Config
  config: {
    ollamaUrl: 'http://localhost:11434',
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 2048,
    systemPromptPreset: 'default',
    customSystemPrompt: 'Tu es Gemma 4, une intelligence artificielle développée par Google DeepMind. Tu es conçue pour être utile, précise, rigoureuse et concise. Tu excelles en raisonnement logique, en programmation et en analyse multimodale.'
  },

  // Performance Telemetry
  telemetry: {
    startTime: 0,
    firstTokenTime: 0,
    totalTokens: 0,
    speed: 0
  },

  // WebLLM Engine Instance
  webLlmEngine: null,
  webLlmLoading: false
};

// Preset system prompts
const SYSTEM_PRESETS = {
  default: 'Tu es Gemma 4, une intelligence artificielle avancée développée par Google DeepMind. Tu es conçue pour être utile, précise, rigoureuse et concise.',
  coder: 'Tu es Gemma 4 en mode Expert Ingénieur Logiciel & Code. Fournis du code moderne, propre, typé et robuste. Explique l\'architecture et les choix algorithmiques clés.',
  reasoner: 'Tu es Gemma 4 en mode Raisonnement Logique Profond. Pour chaque problème, décompose ta réflexion étape par étape en explorant les hypothèses avant de donner la conclusion finale.',
  concise: 'Tu es Gemma 4 en mode Synthétique. Réponds de manière ultra-directe, factuelle et sans verbiage superflu.'
};

// DOM Elements Cache
const DOM = {};

function initDOMElements() {
  DOM.sidebar = document.getElementById('sidebar');
  DOM.btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  DOM.btnNewChat = document.getElementById('btn-new-chat');
  DOM.engineWebgpuBtn = document.getElementById('engine-webgpu-btn');
  DOM.engineOllamaBtn = document.getElementById('engine-ollama-btn');
  DOM.modelSelect = document.getElementById('model-select');
  DOM.modelStatusIndicator = document.getElementById('model-status-indicator');
  DOM.statusText = document.getElementById('status-text');
  
  DOM.modelDownloadBox = document.getElementById('model-download-box');
  DOM.downloadStatusLabel = document.getElementById('download-status-label');
  DOM.downloadPercentage = document.getElementById('download-percentage');
  DOM.downloadProgressBar = document.getElementById('download-progress-bar');
  DOM.downloadDetails = document.getElementById('download-details');

  DOM.chatHistoryList = document.getElementById('chat-history-list');
  DOM.btnOpenSettings = document.getElementById('btn-open-settings');
  DOM.btnExportChat = document.getElementById('btn-export-chat');

  DOM.activeModelName = document.getElementById('active-model-name');
  DOM.activeEngineBadge = document.getElementById('active-engine-badge');
  DOM.hudSpeed = document.getElementById('hud-speed');
  DOM.hudTtft = document.getElementById('hud-ttft');
  DOM.hudTokens = document.getElementById('hud-tokens');

  DOM.messagesContainer = document.getElementById('messages-container');
  DOM.welcomeHero = document.getElementById('welcome-hero');

  DOM.attachmentBar = document.getElementById('attachment-preview-bar');
  DOM.attachedImagePreview = document.getElementById('attached-image-preview');
  DOM.attachmentNameLabel = document.getElementById('attachment-name-label');
  DOM.btnRemoveAttachment = document.getElementById('btn-remove-attachment');
  DOM.btnAttachImage = document.getElementById('btn-attach-image');
  DOM.imageFileInput = document.getElementById('image-file-input');

  DOM.chatForm = document.getElementById('chat-form');
  DOM.userPromptInput = document.getElementById('user-prompt-input');
  DOM.btnToggleThinking = document.getElementById('btn-toggle-thinking');
  DOM.btnSend = document.getElementById('btn-send');
  DOM.gpuBackendInfo = document.getElementById('gpu-backend-info');

  DOM.settingsModal = document.getElementById('settings-modal');
  DOM.btnCloseModal = document.getElementById('btn-close-modal');
  DOM.ollamaUrlInput = document.getElementById('ollama-url-input');
  DOM.btnTestConnection = document.getElementById('btn-test-connection');
  DOM.connectionStatusMsg = document.getElementById('connection-status-msg');
  DOM.temperatureSlider = document.getElementById('temperature-slider');
  DOM.temperatureVal = document.getElementById('temperature-val');
  DOM.topPSlider = document.getElementById('top-p-slider');
  DOM.topPVal = document.getElementById('top-p-val');
  DOM.maxTokensSlider = document.getElementById('max-tokens-slider');
  DOM.maxTokensVal = document.getElementById('max-tokens-val');
  DOM.systemPresetSelect = document.getElementById('system-preset-select');
  DOM.systemPromptText = document.getElementById('system-prompt-text');
  DOM.btnResetSettings = document.getElementById('btn-reset-settings');
  DOM.btnSaveSettings = document.getElementById('btn-save-settings');
}

/* ==========================================================================
   INITIALIZATION
   ========================================================================== */
async function initApp() {
  initDOMElements();
  loadConfig();
  loadConversations();
  setupEventListeners();
  checkWebGPUSupport();
  
  // Set default chat
  if (state.conversations.length === 0) {
    createNewConversation();
  } else {
    loadConversation(state.conversations[0].id);
  }

  // Pre-load WebLLM dynamically
  try {
    webllm = await import('@mlc-ai/web-llm');
    console.log('WebLLM library loaded successfully');
  } catch (err) {
    console.warn('Could not load local @mlc-ai/web-llm, fallback to CDN esm.run', err);
    try {
      webllm = await import('https://esm.run/@mlc-ai/web-llm');
    } catch (cdnErr) {
      console.error('Failed to load WebLLM from CDN', cdnErr);
    }
  }

  updateEngineUI();
}

/* ==========================================================================
   HARDWARE & ENGINE SUPPORT
   ========================================================================== */
async function checkWebGPUSupport() {
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = await (adapter.requestAdapterInfo ? adapter.requestAdapterInfo() : Promise.resolve({}));
        const gpuName = info.description || info.vendor || 'GPU Détecté';
        DOM.gpuBackendInfo.textContent = `WebGPU Accélération : ${gpuName}`;
        DOM.gpuBackendInfo.style.color = '#10b981';
        return true;
      }
    } catch (e) {
      console.warn('WebGPU adapter query failed:', e);
    }
  }
  DOM.gpuBackendInfo.textContent = 'WebGPU non disponible (Activer WebGPU ou utiliser Ollama)';
  DOM.gpuBackendInfo.style.color = '#f59e0b';
  return false;
}

function updateEngineUI() {
  if (state.currentEngine === 'webgpu') {
    DOM.engineWebgpuBtn.classList.add('active');
    DOM.engineOllamaBtn.classList.remove('active');
    DOM.activeEngineBadge.textContent = 'WebGPU In-Browser';
    DOM.activeEngineBadge.style.color = 'var(--accent-cyan)';
    
    // Model Options for WebGPU
    DOM.modelSelect.innerHTML = `
      <optgroup label="Gemma 4 Edge (WebGPU In-Browser)">
        <option value="gemma-2-2b-it-q4f32_1-MLC" selected>Gemma 4 / 2 2B Instruct (4-bit MLC)</option>
        <option value="gemma-2-9b-it-q4f32_1-MLC">Gemma 4 / 2 9B Instruct (4-bit MLC - VRAM 6GB+)</option>
        <option value="Llama-3.2-1B-Instruct-q4f32_1-MLC">Llama 3.2 1B Instruct (Ultra-rapide 800MB)</option>
        <option value="Llama-3.2-3B-Instruct-q4f32_1-MLC">Llama 3.2 3B Instruct (Qualité Edge)</option>
      </optgroup>
    `;
    state.selectedModel = DOM.modelSelect.value;
    updateActiveModelDisplay();
  } else {
    DOM.engineWebgpuBtn.classList.remove('active');
    DOM.engineOllamaBtn.classList.add('active');
    DOM.activeEngineBadge.textContent = 'Ollama Localhost';
    DOM.activeEngineBadge.style.color = 'var(--accent-emerald)';
    
    // Fetch models from Ollama
    fetchOllamaModels();
  }
}

async function fetchOllamaModels() {
  DOM.statusText.textContent = 'Détection...';
  DOM.modelStatusIndicator.className = 'status-indicator loading';
  
  try {
    const res = await fetch(`${state.config.ollamaUrl}/api/tags`, { method: 'GET' });
    if (!res.ok) throw new Error('Impossible de joindre Ollama');
    const data = await res.json();
    
    if (data.models && data.models.length > 0) {
      let gemmaGroup = '';
      let otherGroup = '';
      
      data.models.forEach(m => {
        const isGemma = m.name.toLowerCase().includes('gemma');
        const opt = `<option value="${m.name}" ${isGemma ? 'selected' : ''}>${m.name} (${formatSize(m.size)})</option>`;
        if (isGemma) gemmaGroup += opt;
        else otherGroup += opt;
      });

      DOM.modelSelect.innerHTML = `
        ${gemmaGroup ? `<optgroup label="Modèles Gemma Locaux">${gemmaGroup}</optgroup>` : ''}
        ${otherGroup ? `<optgroup label="Autres Modèles Ollama">${otherGroup}</optgroup>` : ''}
      `;
      state.selectedModel = DOM.modelSelect.value;
      DOM.statusText.textContent = 'Connecté';
      DOM.modelStatusIndicator.className = 'status-indicator ready';
    } else {
      DOM.modelSelect.innerHTML = `<option value="gemma4">gemma4 (non téléchargé)</option>`;
      DOM.statusText.textContent = 'Aucun modèle';
      DOM.modelStatusIndicator.className = 'status-indicator error';
    }
  } catch (err) {
    console.warn('Ollama not reachable, using default list:', err);
    DOM.modelSelect.innerHTML = `
      <optgroup label="Ollama (Non connecté - Lancez 'ollama serve')">
        <option value="gemma4:latest">gemma4:latest</option>
        <option value="gemma4:e2b">gemma4:e2b (Edge 2B)</option>
        <option value="gemma4:e4b">gemma4:e4b (Edge 4B)</option>
        <option value="gemma4:12b">gemma4:12b</option>
        <option value="gemma2:2b">gemma2:2b</option>
        <option value="gemma2:9b">gemma2:9b</option>
      </optgroup>
    `;
    DOM.statusText.textContent = 'Déconnecté';
    DOM.modelStatusIndicator.className = 'status-indicator error';
  }
  updateActiveModelDisplay();
}

function formatSize(bytes) {
  if (!bytes) return '';
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(1)} GB`;
}

function updateActiveModelDisplay() {
  const modelText = DOM.modelSelect.options[DOM.modelSelect.selectedIndex]?.text || state.selectedModel;
  DOM.activeModelName.textContent = modelText.split('(')[0].trim();
}

/* ==========================================================================
   WEBGPU ENGINE LOADER (@mlc-ai/web-llm)
   ========================================================================== */
async function ensureWebLLMEngine() {
  if (state.webLlmEngine && state.webLlmEngine.modelId === state.selectedModel) {
    return state.webLlmEngine;
  }

  if (!webllm) {
    throw new Error('La bibliothèque WebLLM n\'a pas pu être chargée.');
  }

  DOM.modelDownloadBox.classList.remove('hidden');
  DOM.modelStatusIndicator.className = 'status-indicator loading';
  DOM.statusText.textContent = 'Chargement';

  const initProgressCallback = (report) => {
    DOM.downloadDetails.textContent = report.text;
    const progress = Math.round(report.progress * 100);
    DOM.downloadPercentage.textContent = `${progress}%`;
    DOM.downloadProgressBar.style.width = `${progress}%`;
    
    if (progress >= 100) {
      DOM.downloadStatusLabel.textContent = 'Modèle prêt en mémoire GPU !';
    } else {
      DOM.downloadStatusLabel.textContent = 'Téléchargement & compilation shader...';
    }
  };

  try {
    state.webLlmLoading = true;
    const engine = await webllm.CreateMLCEngine(state.selectedModel, {
      initProgressCallback
    });
    engine.modelId = state.selectedModel;
    state.webLlmEngine = engine;
    state.webLlmLoading = false;

    DOM.modelStatusIndicator.className = 'status-indicator ready';
    DOM.statusText.textContent = 'Prêt';
    setTimeout(() => {
      DOM.modelDownloadBox.classList.add('hidden');
    }, 2500);

    return engine;
  } catch (err) {
    state.webLlmLoading = false;
    DOM.modelStatusIndicator.className = 'status-indicator error';
    DOM.statusText.textContent = 'Erreur';
    DOM.downloadStatusLabel.textContent = 'Échec d\'initialisation WebGPU';
    DOM.downloadDetails.textContent = err.message || 'Erreur lors du chargement des poids.';
    throw err;
  }
}

/* ==========================================================================
   CHAT INFERENCE & STREAMING CONTROLLER
   ========================================================================== */
async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  
  const text = DOM.userPromptInput.value.trim();
  if ((!text && !state.attachedImage) || state.isGenerating) return;

  // Add User Message
  const currentChat = getCurrentConversation();
  const userMsg = {
    role: 'user',
    content: text,
    image: state.attachedImage ? state.attachedImage.base64 : null,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  currentChat.messages.push(userMsg);
  
  // Clear input & attachment
  DOM.userPromptInput.value = '';
  DOM.userPromptInput.style.height = 'auto';
  clearAttachment();
  saveConversations();

  // If first message, update chat title
  if (currentChat.messages.length === 1 && text) {
    currentChat.title = text.slice(0, 32) + (text.length > 32 ? '...' : '');
    renderHistoryList();
  }

  // Render User Message in UI
  renderMessage(userMsg, currentChat.messages.length - 1);
  hideWelcomeHero();
  scrollToBottom();

  // Prepare Assistant placeholder
  const assistantMsgIndex = currentChat.messages.length;
  const assistantMsg = {
    role: 'assistant',
    content: '',
    thinking: '',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };
  currentChat.messages.push(assistantMsg);

  // Set Generating state
  setGenerating(true);
  resetTelemetry();

  try {
    if (state.currentEngine === 'webgpu') {
      await streamWebLLM(currentChat, assistantMsgIndex);
    } else {
      await streamOllama(currentChat, assistantMsgIndex);
    }
  } catch (err) {
    console.error('Inference error:', err);
    if (err.name !== 'AbortError') {
      assistantMsg.content += `\n\n*(Erreur d'inférence : ${err.message || 'Connexion interrompue'})*`;
      updateAssistantMessageUI(assistantMsgIndex, assistantMsg);
    }
  } finally {
    setGenerating(false);
    saveConversations();
    finalizeMessageUI(assistantMsgIndex);
  }
}

/* --- Stream via WebGPU (WebLLM) --- */
async function streamWebLLM(conversation, msgIndex) {
  const engine = await ensureWebLLMEngine();
  state.abortController = new AbortController();

  // Format messages for MLC Engine
  const formattedMessages = [
    { role: 'system', content: state.config.customSystemPrompt }
  ];

  conversation.messages.slice(0, -1).forEach(m => {
    formattedMessages.push({
      role: m.role,
      content: m.content
    });
  });

  const completion = await engine.chat.completions.create({
    messages: formattedMessages,
    temperature: parseFloat(state.config.temperature),
    top_p: parseFloat(state.config.top_p),
    max_tokens: parseInt(state.config.max_tokens),
    stream: true
  });

  let fullResponse = '';
  let fullThinking = '';
  let inThinkingTag = false;

  for await (const chunk of completion) {
    if (state.abortController?.signal.aborted) break;

    const delta = chunk.choices[0]?.delta?.content || '';
    if (!delta) continue;

    trackTokenTelemetry();

    // Check for Gemma 4 Thinking Mode tags: <thought>, <thinking>, </thought>, etc.
    const parsed = processThinkingTokens(delta, inThinkingTag, fullThinking, fullResponse);
    inThinkingTag = parsed.inThinkingTag;
    fullThinking = parsed.thinking;
    fullResponse = parsed.content;

    conversation.messages[msgIndex].content = fullResponse;
    conversation.messages[msgIndex].thinking = fullThinking;

    updateAssistantMessageUI(msgIndex, conversation.messages[msgIndex], inThinkingTag);
    scrollToBottom();
  }
}

/* --- Stream via Ollama Localhost API --- */
async function streamOllama(conversation, msgIndex) {
  state.abortController = new AbortController();

  const formattedMessages = [
    { role: 'system', content: state.config.customSystemPrompt }
  ];

  conversation.messages.slice(0, -1).forEach(m => {
    const msgObj = { role: m.role, content: m.content };
    if (m.image) {
      // Ollama expects array of base64 strings (without data:image prefix)
      const cleanBase64 = m.image.replace(/^data:image\/[a-z]+;base64,/, '');
      msgObj.images = [cleanBase64];
    }
    formattedMessages.push(msgObj);
  });

  const payload = {
    model: state.selectedModel,
    messages: formattedMessages,
    stream: true,
    options: {
      temperature: parseFloat(state.config.temperature),
      top_p: parseFloat(state.config.top_p),
      num_predict: parseInt(state.config.max_tokens)
    }
  };

  const response = await fetch(`${state.config.ollamaUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: state.abortController.signal
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama HTTP ${response.status}: ${errorText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullResponse = '';
  let fullThinking = '';
  let inThinkingTag = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep last incomplete line

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const json = JSON.parse(line);
        const token = json.message?.content || '';
        
        if (token) {
          trackTokenTelemetry();

          const parsed = processThinkingTokens(token, inThinkingTag, fullThinking, fullResponse);
          inThinkingTag = parsed.inThinkingTag;
          fullThinking = parsed.thinking;
          fullResponse = parsed.content;

          conversation.messages[msgIndex].content = fullResponse;
          conversation.messages[msgIndex].thinking = fullThinking;

          updateAssistantMessageUI(msgIndex, conversation.messages[msgIndex], inThinkingTag);
          scrollToBottom();
        }
      } catch (e) {
        console.warn('JSON parse error in Ollama stream chunk:', line);
      }
    }
  }
}

/* Helper to process Gemma 4 thinking tags (<thought>, <thinking>, </thought>) */
function processThinkingTokens(token, inThinkingTag, currentThinking, currentContent) {
  let text = token;
  let inTag = inThinkingTag;
  let thinking = currentThinking;
  let content = currentContent;

  // Detect opening tag
  if (text.includes('<thought>') || text.includes('<thinking>')) {
    inTag = true;
    text = text.replace('<thought>', '').replace('<thinking>', '');
  }

  // Detect closing tag
  if (text.includes('</thought>') || text.includes('</thinking>')) {
    const parts = text.split(/<\/thought>|<\/thinking>/);
    thinking += parts[0];
    content += parts[1] || '';
    inTag = false;
    return { inThinkingTag: inTag, thinking, content };
  }

  if (inTag) {
    thinking += text;
  } else {
    content += text;
  }

  return { inThinkingTag: inTag, thinking, content };
}

/* ==========================================================================
   TELEMETRY & HUD
   ========================================================================== */
function resetTelemetry() {
  state.telemetry.startTime = performance.now();
  state.telemetry.firstTokenTime = 0;
  state.telemetry.totalTokens = 0;
  DOM.hudTokens.textContent = '0';
  DOM.hudTtft.innerHTML = `-- <small>ms</small>`;
  DOM.hudSpeed.innerHTML = `0.0 <small>tok/s</small>`;
}

function trackTokenTelemetry() {
  const now = performance.now();
  state.telemetry.totalTokens++;

  if (state.telemetry.totalTokens === 1) {
    state.telemetry.firstTokenTime = Math.round(now - state.telemetry.startTime);
    DOM.hudTtft.innerHTML = `${state.telemetry.firstTokenTime} <small>ms</small>`;
  }

  const elapsedSec = (now - state.telemetry.startTime) / 1000;
  if (elapsedSec > 0.3) {
    const tokPerSec = (state.telemetry.totalTokens / elapsedSec).toFixed(1);
    DOM.hudSpeed.innerHTML = `${tokPerSec} <small>tok/s</small>`;
  }

  DOM.hudTokens.textContent = state.telemetry.totalTokens;
}

function setGenerating(isGen) {
  state.isGenerating = isGen;
  if (isGen) {
    DOM.btnSend.querySelector('.send-icon').classList.add('hidden');
    DOM.btnSend.querySelector('.stop-icon').classList.remove('hidden');
    DOM.btnSend.title = 'Arrêter la génération';
  } else {
    DOM.btnSend.querySelector('.send-icon').classList.remove('hidden');
    DOM.btnSend.querySelector('.stop-icon').classList.add('hidden');
    DOM.btnSend.title = 'Envoyer le message';
  }
}

function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
  }
  setGenerating(false);
}

/* ==========================================================================
   UI RENDERING (MESSAGES, THINKING, MARKDOWN, CODE)
   ========================================================================== */
function renderMessage(msg, index) {
  const row = document.createElement('div');
  row.className = `message-row ${msg.role}`;
  row.dataset.index = index;

  const isUser = msg.role === 'user';
  const avatarText = isUser ? 'U' : '✦';
  const senderName = isUser ? 'Vous' : `Gemma 4 (${state.selectedModel.split(':')[0]})`;

  let attachedImageHtml = '';
  if (msg.image) {
    attachedImageHtml = `<img src="${msg.image}" class="message-attached-image" alt="Image analysée" />`;
  }

  let thinkingHtml = '';
  if (msg.thinking) {
    thinkingHtml = `
      <div class="thinking-card open">
        <div class="thinking-header" onclick="this.parentElement.classList.toggle('open'); this.parentElement.classList.toggle('closed');">
          <div class="thinking-title-group">
            <span class="thinking-spinner"></span>
            <span>Processus de réflexion (Gemma 4)</span>
          </div>
          <span class="thinking-chevron">▲</span>
        </div>
        <div class="thinking-body">${escapeHtml(msg.thinking)}</div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="message-avatar">${avatarText}</div>
    <div class="message-body">
      <div class="message-sender">
        <span>${senderName}</span>
        <span class="message-time">${msg.timestamp || ''}</span>
      </div>
      <div class="message-content">
        ${attachedImageHtml}
        ${thinkingHtml}
        <div class="message-text">${formatMarkdown(msg.content)}</div>
      </div>
    </div>
  `;

  DOM.messagesContainer.appendChild(row);
  attachCodeCopyButtons(row);
}

function updateAssistantMessageUI(index, msg, isThinking = false) {
  let row = DOM.messagesContainer.querySelector(`.message-row[data-index="${index}"]`);
  if (!row) {
    renderMessage(msg, index);
    row = DOM.messagesContainer.querySelector(`.message-row[data-index="${index}"]`);
  }

  const contentContainer = row.querySelector('.message-content');
  
  // Manage Thinking Box
  let thinkingCard = contentContainer.querySelector('.thinking-card');
  if (msg.thinking) {
    if (!thinkingCard) {
      thinkingCard = document.createElement('div');
      thinkingCard.className = 'thinking-card open';
      thinkingCard.innerHTML = `
        <div class="thinking-header">
          <div class="thinking-title-group">
            <span class="thinking-spinner"></span>
            <span>Processus de réflexion (Gemma 4)</span>
          </div>
          <span class="thinking-chevron">▲</span>
        </div>
        <div class="thinking-body"></div>
      `;
      thinkingCard.querySelector('.thinking-header').addEventListener('click', () => {
        thinkingCard.classList.toggle('open');
        thinkingCard.classList.toggle('closed');
      });
      contentContainer.insertBefore(thinkingCard, contentContainer.querySelector('.message-text'));
    }
    thinkingCard.querySelector('.thinking-body').textContent = msg.thinking;
    
    // Stop spinner if thinking completed
    if (!isThinking) {
      const spinner = thinkingCard.querySelector('.thinking-spinner');
      if (spinner) spinner.style.display = 'none';
    }
  }

  const textDiv = contentContainer.querySelector('.message-text');
  textDiv.innerHTML = formatMarkdown(msg.content) + '<span class="streaming-cursor"></span>';
  attachCodeCopyButtons(row);
}

function finalizeMessageUI(index) {
  const row = DOM.messagesContainer.querySelector(`.message-row[data-index="${index}"]`);
  if (!row) return;

  const cursor = row.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();

  const spinner = row.querySelector('.thinking-spinner');
  if (spinner) spinner.style.display = 'none';

  // Automatically fold thinking block to keep answers clean
  const thinkingCard = row.querySelector('.thinking-card');
  if (thinkingCard) {
    thinkingCard.classList.remove('open');
    thinkingCard.classList.add('closed');
  }
}

/* Minimalist & High-Performance Markdown Parser */
function formatMarkdown(text) {
  if (!text) return '';

  let html = escapeHtml(text);

  // Fenced Code blocks
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const language = lang || 'code';
    return `
      <pre><div class="code-header"><span>${language}</span><button class="btn-copy-code" data-code="${encodeURIComponent(code)}">Copier</button></div><code>${code.trim()}</code></pre>
    `;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Lists
  html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Paragraphs & linebreaks
  html = html.split('\n\n').map(p => {
    if (p.startsWith('<pre') || p.startsWith('<h') || p.startsWith('<ul')) return p;
    return `<p>${p.replace(/\n/g, '<br/>')}</p>`;
  }).join('');

  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function attachCodeCopyButtons(container) {
  container.querySelectorAll('.btn-copy-code').forEach(btn => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = 'true';
    btn.addEventListener('click', () => {
      const code = decodeURIComponent(btn.dataset.code);
      navigator.clipboard.writeText(code).then(() => {
        const originalText = btn.textContent;
        btn.textContent = '✓ Copié !';
        btn.style.color = '#10b981';
        setTimeout(() => {
          btn.textContent = originalText;
          btn.style.color = '';
        }, 2000);
      });
    });
  });
}

function scrollToBottom() {
  DOM.messagesContainer.scrollTop = DOM.messagesContainer.scrollHeight;
}

function hideWelcomeHero() {
  if (DOM.welcomeHero) DOM.welcomeHero.classList.add('hidden');
}

function showWelcomeHero() {
  if (DOM.welcomeHero) DOM.welcomeHero.classList.remove('hidden');
}

/* ==========================================================================
   CONVERSATION MANAGEMENT
   ========================================================================== */
function loadConversations() {
  try {
    const raw = localStorage.getItem('gemma4_conversations');
    state.conversations = raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Failed to parse conversations from storage', e);
    state.conversations = [];
  }
  renderHistoryList();
}

function saveConversations() {
  try {
    localStorage.setItem('gemma4_conversations', JSON.stringify(state.conversations));
  } catch (e) {
    console.error('Failed to save conversations to storage', e);
  }
}

function createNewConversation() {
  const newChat = {
    id: 'chat_' + Date.now(),
    title: 'Nouvelle conversation',
    createdAt: new Date().toISOString(),
    messages: []
  };
  state.conversations.unshift(newChat);
  saveConversations();
  renderHistoryList();
  loadConversation(newChat.id);
}

function getCurrentConversation() {
  return state.conversations.find(c => c.id === state.activeConversationId) || state.conversations[0];
}

function loadConversation(chatId) {
  state.activeConversationId = chatId;
  renderHistoryList();

  // Clear message feed
  DOM.messagesContainer.innerHTML = '';
  const chat = getCurrentConversation();

  if (!chat || chat.messages.length === 0) {
    DOM.messagesContainer.appendChild(DOM.welcomeHero);
    showWelcomeHero();
  } else {
    hideWelcomeHero();
    chat.messages.forEach((m, idx) => renderMessage(m, idx));
    scrollToBottom();
  }
}

function deleteConversation(chatId, e) {
  if (e) e.stopPropagation();
  state.conversations = state.conversations.filter(c => c.id !== chatId);
  saveConversations();

  if (state.activeConversationId === chatId) {
    if (state.conversations.length > 0) {
      loadConversation(state.conversations[0].id);
    } else {
      createNewConversation();
    }
  } else {
    renderHistoryList();
  }
}

function renderHistoryList() {
  DOM.chatHistoryList.innerHTML = '';
  state.conversations.forEach(c => {
    const item = document.createElement('div');
    item.className = `history-item ${c.id === state.activeConversationId ? 'active' : ''}`;
    item.innerHTML = `
      <span class="history-item-title">${escapeHtml(c.title)}</span>
      <button class="history-item-delete" title="Supprimer">✕</button>
    `;
    item.addEventListener('click', () => loadConversation(c.id));
    item.querySelector('.history-item-delete').addEventListener('click', (e) => deleteConversation(c.id, e));
    DOM.chatHistoryList.appendChild(item);
  });
}

function exportCurrentChat() {
  const chat = getCurrentConversation();
  if (!chat || chat.messages.length === 0) return alert('Aucun message à exporter.');

  let md = `# ${chat.title}\n*Date: ${chat.createdAt}*\n*Modèle: ${state.selectedModel}*\n\n---\n\n`;
  chat.messages.forEach(m => {
    const roleName = m.role === 'user' ? '### 👤 Utilisateur' : '### ✦ Gemma 4';
    md += `${roleName} (${m.timestamp || ''})\n\n`;
    if (m.thinking) {
      md += `> **Raisonnement Gemma 4 :**\n> ${m.thinking.replace(/\n/g, '\n> ')}\n\n`;
    }
    md += `${m.content}\n\n---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${chat.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ==========================================================================
   MULTIMODAL ATTACHMENT
   ========================================================================== */
function handleImageUpload(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    state.attachedImage = {
      base64: e.target.result,
      name: file.name
    };
    DOM.attachedImagePreview.src = e.target.result;
    DOM.attachmentNameLabel.textContent = file.name;
    DOM.attachmentBar.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function clearAttachment() {
  state.attachedImage = null;
  DOM.attachmentBar.classList.add('hidden');
  DOM.attachedImagePreview.src = '';
  DOM.imageFileInput.value = '';
}

/* ==========================================================================
   CONFIG & SETTINGS MODAL
   ========================================================================== */
function loadConfig() {
  try {
    const saved = localStorage.getItem('gemma4_config');
    if (saved) {
      state.config = { ...state.config, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.error('Error loading config:', e);
  }
}

function saveConfig() {
  try {
    localStorage.setItem('gemma4_config', JSON.stringify(state.config));
  } catch (e) {
    console.error('Error saving config:', e);
  }
}

function openSettingsModal() {
  DOM.ollamaUrlInput.value = state.config.ollamaUrl;
  DOM.temperatureSlider.value = state.config.temperature;
  DOM.temperatureVal.textContent = state.config.temperature;
  DOM.topPSlider.value = state.config.top_p;
  DOM.topPVal.textContent = state.config.top_p;
  DOM.maxTokensSlider.value = state.config.max_tokens;
  DOM.maxTokensVal.textContent = state.config.max_tokens;
  DOM.systemPresetSelect.value = state.config.systemPromptPreset;
  DOM.systemPromptText.value = state.config.customSystemPrompt;
  DOM.connectionStatusMsg.classList.add('hidden');
  DOM.settingsModal.classList.remove('hidden');
}

function closeSettingsModal() {
  DOM.settingsModal.classList.add('hidden');
}

async function testOllamaConnection() {
  const url = DOM.ollamaUrlInput.value.trim().replace(/\/$/, '');
  DOM.btnTestConnection.textContent = 'Test...';
  
  try {
    const res = await fetch(`${url}/api/tags`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const count = data.models ? data.models.length : 0;
    
    DOM.connectionStatusMsg.className = 'connection-status-msg success';
    DOM.connectionStatusMsg.textContent = `✓ Connexion réussie ! ${count} modèle(s) détecté(s).`;
    DOM.connectionStatusMsg.classList.remove('hidden');
  } catch (err) {
    DOM.connectionStatusMsg.className = 'connection-status-msg error';
    DOM.connectionStatusMsg.textContent = `✕ Échec : Impossible de joindre ${url} (${err.message}). Vérifiez CORS et 'ollama serve'.`;
    DOM.connectionStatusMsg.classList.remove('hidden');
  } finally {
    DOM.btnTestConnection.textContent = 'Tester';
  }
}

/* ==========================================================================
   EVENT LISTENERS SETUP
   ========================================================================== */
function setupEventListeners() {
  // Mobile Sidebar Toggle
  DOM.btnToggleSidebar.addEventListener('click', () => {
    DOM.sidebar.classList.toggle('open');
  });

  // New Chat
  DOM.btnNewChat.addEventListener('click', () => {
    createNewConversation();
    if (window.innerWidth < 768) DOM.sidebar.classList.remove('open');
  });

  // Engine Switching
  DOM.engineWebgpuBtn.addEventListener('click', () => {
    state.currentEngine = 'webgpu';
    updateEngineUI();
  });
  DOM.engineOllamaBtn.addEventListener('click', () => {
    state.currentEngine = 'ollama';
    updateEngineUI();
  });

  // Model Selection
  DOM.modelSelect.addEventListener('change', () => {
    state.selectedModel = DOM.modelSelect.value;
    updateActiveModelDisplay();
  });

  // Form Submit
  DOM.chatForm.addEventListener('submit', (e) => {
    if (state.isGenerating) {
      e.preventDefault();
      stopGeneration();
    } else {
      handleChatSubmit(e);
    }
  });

  // Auto-resize input & enter key listener
  DOM.userPromptInput.addEventListener('input', () => {
    DOM.userPromptInput.style.height = 'auto';
    DOM.userPromptInput.style.height = `${Math.min(DOM.userPromptInput.scrollHeight, 160)}px`;
  });

  DOM.userPromptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatSubmit();
    }
  });

  // Thinking Mode Toggle Button
  DOM.btnToggleThinking.addEventListener('click', () => {
    state.thinkingModeEnabled = !state.thinkingModeEnabled;
    DOM.btnToggleThinking.classList.toggle('active', state.thinkingModeEnabled);
  });

  // Multimodal Attachments
  DOM.btnAttachImage.addEventListener('click', () => DOM.imageFileInput.click());
  DOM.imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  });
  DOM.btnRemoveAttachment.addEventListener('click', clearAttachment);

  // Drag and Drop Images on App
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  });

  // Quick Prompt Cards
  document.querySelectorAll('.quick-prompt-card').forEach(card => {
    card.addEventListener('click', () => {
      DOM.userPromptInput.value = card.dataset.prompt;
      DOM.userPromptInput.focus();
      handleChatSubmit();
    });
  });

  // Export Chat
  DOM.btnExportChat.addEventListener('click', exportCurrentChat);

  // Settings Modal Controls
  DOM.btnOpenSettings.addEventListener('click', openSettingsModal);
  DOM.btnCloseModal.addEventListener('click', closeSettingsModal);
  DOM.btnTestConnection.addEventListener('click', testOllamaConnection);

  DOM.temperatureSlider.addEventListener('input', (e) => {
    DOM.temperatureVal.textContent = e.target.value;
  });
  DOM.topPSlider.addEventListener('input', (e) => {
    DOM.topPVal.textContent = e.target.value;
  });
  DOM.maxTokensSlider.addEventListener('input', (e) => {
    DOM.maxTokensVal.textContent = e.target.value;
  });

  DOM.systemPresetSelect.addEventListener('change', (e) => {
    const preset = e.target.value;
    if (SYSTEM_PRESETS[preset]) {
      DOM.systemPromptText.value = SYSTEM_PRESETS[preset];
    }
  });

  DOM.btnResetSettings.addEventListener('click', () => {
    DOM.temperatureSlider.value = 0.7;
    DOM.temperatureVal.textContent = '0.7';
    DOM.topPSlider.value = 0.9;
    DOM.topPVal.textContent = '0.9';
    DOM.maxTokensSlider.value = 2048;
    DOM.maxTokensVal.textContent = '2048';
    DOM.systemPresetSelect.value = 'default';
    DOM.systemPromptText.value = SYSTEM_PRESETS.default;
  });

  DOM.btnSaveSettings.addEventListener('click', () => {
    state.config.ollamaUrl = DOM.ollamaUrlInput.value.trim().replace(/\/$/, '');
    state.config.temperature = parseFloat(DOM.temperatureSlider.value);
    state.config.top_p = parseFloat(DOM.topPSlider.value);
    state.config.max_tokens = parseInt(DOM.maxTokensSlider.value);
    state.config.systemPromptPreset = DOM.systemPresetSelect.value;
    state.config.customSystemPrompt = DOM.systemPromptText.value.trim();
    saveConfig();
    closeSettingsModal();
    if (state.currentEngine === 'ollama') {
      fetchOllamaModels();
    }
  });
}

// Start Application on DOM Load
window.addEventListener('DOMContentLoaded', initApp);
