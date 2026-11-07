const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const chatStream = document.getElementById('chatStream');
const welcomePanel = document.getElementById('welcomePanel');
const sendButton = document.getElementById('sendButton');
const charCount = document.getElementById('charCount');
const statusPill = document.getElementById('modelStatus');
const statusText = document.getElementById('statusText');
const aboutDialog = document.getElementById('aboutDialog');

function updateCounter() {
  charCount.textContent = `${messageInput.value.length} / 2000`;
  messageInput.style.height = 'auto';
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 145)}px`;
}

function addMessage(role, text, metadata = {}) {
  const article = document.createElement('article');
  article.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'YOU' : '</>';

  const body = document.createElement('div');
  body.className = 'message-body';
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = role === 'user' ? 'YOU' : 'CODEMENTOR';
  const content = document.createElement('div');
  content.className = 'message-text';
  content.textContent = text;
  body.append(meta, content);

  if (role === 'assistant' && metadata.source) {
    const tools = document.createElement('div');
    tools.className = 'message-tools';
    const tag = document.createElement('span');
    tag.className = 'source-tag';
    tag.textContent = metadata.source;
    tools.append(tag);
    if (metadata.matched_question) {
      const match = document.createElement('span');
      match.className = 'match-tag';
      match.textContent = `Related to: ${metadata.matched_question}`;
      tools.append(match);
    }
    body.append(tools);

    if (metadata.model_output) {
      const details = document.createElement('details');
      details.className = 'model-output';
      const summary = document.createElement('summary');
      summary.textContent = 'View raw output from your trained MiniGPT model';
      const pre = document.createElement('pre');
      pre.textContent = metadata.model_output;
      details.append(summary, pre);
      body.append(details);
    }
  }

  article.append(avatar, body);
  chatStream.append(article);
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return article;
}

function showTyping() {
  const article = addMessage('assistant', 'Generating a local response…');
  article.querySelector('.message-text').classList.add('typing');
  return article;
}

async function sendMessage(text) {
  const question = text.trim();
  if (!question || sendButton.disabled) return;
  welcomePanel.hidden = true;
  addMessage('user', question);
  messageInput.value = '';
  updateCounter();
  sendButton.disabled = true;
  const typing = showTyping();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question })
    });
    const result = await response.json();
    typing.remove();
    if (!response.ok) {
      addMessage('assistant', result.error || 'Something went wrong.');
    } else {
      addMessage('assistant', result.answer, result);
    }
  } catch (error) {
    typing.remove();
    addMessage('assistant', 'Could not reach the local Flask server. Check that app.py is running in your terminal.');
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
  }
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener('input', updateCounter);
messageInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    const prompt = button.dataset.prompt || '';
    messageInput.value = prompt;
    updateCounter();
    messageInput.focus();
    if (button.classList.contains('starter-card')) chatForm.requestSubmit();
  });
});

document.getElementById('newChatButton').addEventListener('click', () => {
  chatStream.replaceChildren();
  welcomePanel.hidden = false;
  messageInput.value = '';
  updateCounter();
  messageInput.focus();
});

document.getElementById('aboutButton').addEventListener('click', () => aboutDialog.showModal());
document.getElementById('closeDialog').addEventListener('click', () => aboutDialog.close());
aboutDialog.addEventListener('click', (event) => {
  if (event.target === aboutDialog) aboutDialog.close();
});

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    if (status.model_loaded) {
      statusPill.classList.add('ready');
      statusText.textContent = `MiniGPT ready · ${status.device}`;
    } else {
      statusText.textContent = `Knowledge mode · ${status.dataset_examples} examples`;
    }
  } catch (_error) {
    statusText.textContent = 'Local status unavailable';
  }
}

updateCounter();
loadStatus();
