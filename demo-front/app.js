const API_BASE = 'http://localhost:9621';
const API_URL = `${API_BASE}/query/stream`;
const STATIC_URL = `${API_BASE}/static`;

const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const quickActions = document.querySelectorAll('.quick-action');
let typingIndicator = null;

// Multi-turn conversation history
const conversationHistory = [];
const MAX_HISTORY_TURNS = 3; // Keep last 3 turns (6 messages)

function buildContextQuery(question) {
    if (conversationHistory.length === 0) {
        return question;
    }

    const recentHistory = conversationHistory.slice(-MAX_HISTORY_TURNS * 2);
    const contextParts = ['Previous conversation:'];

    for (const msg of recentHistory) {
        const role = msg.role === 'user' ? 'User' : 'Assistant';
        const content = msg.content.length > 500
            ? msg.content.substring(0, 500) + '...'
            : msg.content;
        contextParts.push(`${role}: ${content}`);
    }

    contextParts.push(`\nCurrent question: ${question}`);
    return contextParts.join('\n');
}

function addToHistory(role, content) {
    conversationHistory.push({ role, content });
    // Trim history if too long
    while (conversationHistory.length > MAX_HISTORY_TURNS * 2) {
        conversationHistory.shift();
    }
}

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Send message on Enter (Shift+Enter for new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener('click', sendMessage);

// Quick actions
quickActions.forEach(btn => {
    btn.addEventListener('click', () => {
        messageInput.value = btn.dataset.query;
        sendMessage();
    });
});

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? 'You' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMessage(content);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
}

// Configure marked
marked.setOptions({
    breaks: true,
    gfm: true,
});

function formatMessage(text) {
    // Pre-process images to add click handler and static URL
    let processed = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
        const imgSrc = src.startsWith('http') ? src : `${STATIC_URL}/${src}`;
        return `![${alt}](${imgSrc})`;
    });

    // Render markdown
    let html = marked.parse(processed);

    // Post-process: add click handler to images
    html = html.replace(/<img([^>]*)src="([^"]+)"([^>]*)>/g,
        '<img$1src="$2"$3 class="chat-image" onclick="openImageModal(\'$2\')">'
    );

    return html;
}

function openImageModal(src) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    modal.style.display = 'flex';
    modalImg.src = src;
}

function closeImageModal() {
    document.getElementById('imageModal').style.display = 'none';
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function setLoading(loading) {
    sendButton.disabled = loading;
    messageInput.disabled = loading;

    if (loading) {
        typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator active';
        typingIndicator.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        messagesContainer.appendChild(typingIndicator);
        scrollToBottom();
    } else {
        if (typingIndicator && typingIndicator.parentNode) {
            typingIndicator.remove();
            typingIndicator = null;
        }
    }
}

function createStreamingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);

    return contentDiv;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    addToHistory('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    setLoading(true);

    try {
        const contextQuery = buildContextQuery(message);

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: contextQuery, mode: 'hybrid' }),
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        let streamTarget = null;
        let fullText = '';
        let referencedImages = [];
        let firstChunkReceived = false;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const json = JSON.parse(line);
                    if (json.referenced_images) {
                        referencedImages = json.referenced_images;
                        console.log('Referenced images:', referencedImages);
                    }
                    if (json.response) {
                        // 첫 응답 청크가 오면 인디케이터 제거 후 메시지 영역 생성
                        if (!firstChunkReceived) {
                            firstChunkReceived = true;
                            if (typingIndicator && typingIndicator.parentNode) {
                                typingIndicator.remove();
                                typingIndicator = null;
                            }
                            streamTarget = createStreamingMessage();
                        }
                        fullText += json.response;
                        if (streamTarget) {
                            streamTarget.innerHTML = formatMessage(fullText);
                            scrollToBottom();
                        }
                    }
                } catch (e) {
                    // Skip invalid JSON
                }
            }
        }

        // Process remaining buffer
        if (buffer.trim()) {
            try {
                const json = JSON.parse(buffer);
                if (json.response) {
                    if (!firstChunkReceived) {
                        firstChunkReceived = true;
                        if (typingIndicator && typingIndicator.parentNode) {
                            typingIndicator.remove();
                            typingIndicator = null;
                        }
                        streamTarget = createStreamingMessage();
                    }
                    fullText += json.response;
                    if (streamTarget) {
                        streamTarget.innerHTML = formatMessage(fullText);
                    }
                }
            } catch (e) {}
        }

        // 응답이 없었으면 인디케이터 제거
        if (!firstChunkReceived && typingIndicator && typingIndicator.parentNode) {
            typingIndicator.remove();
            typingIndicator = null;
        }

        // 참조된 이미지가 있으면 추가 표시
        if (referencedImages.length > 0 && streamTarget) {
            console.log('Processing images:', referencedImages);
            let imageHtml = '<div class="reference-images"><p><strong>참조 이미지:</strong></p><div class="reference-images-grid">';
            for (const imgPath of referencedImages) {
                // 경로에서 output/ 이후 부분만 추출
                let relativePath = imgPath;
                const outputIndex = imgPath.indexOf('/output/');
                if (outputIndex !== -1) {
                    relativePath = imgPath.substring(outputIndex + '/output/'.length);
                }
                const imgSrc = `${STATIC_URL}/${relativePath}`;
                console.log('Image path:', imgPath, '-> URL:', imgSrc);
                imageHtml += `<img src="${imgSrc}" alt="참조 이미지" class="chat-image" onclick="openImageModal('${imgSrc}')" onerror="console.error('Failed to load:', '${imgSrc}'); this.style.display='none'" />`;
            }
            imageHtml += '</div></div>';
            streamTarget.innerHTML += imageHtml;
            scrollToBottom();
        }

        // Save assistant response to history
        if (fullText) {
            addToHistory('assistant', fullText);
        }

    } catch (error) {
        console.error('Error:', error);
        addMessage('죄송합니다. 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.');
    } finally {
        setLoading(false);
    }
}
