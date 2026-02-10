"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "../ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Input } from "../ui/input";

export default function ChatbotAssistant({ userId }) {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your mental health companion. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const synthRef = useRef(null);

  // Load conversations from localStorage on mount
  useEffect(() => {
    if (!userId) return;
    
    const savedConversations = localStorage.getItem(
      `chatbot-conversations-${userId}`
    );
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setConversations(parsed);
          // Parse dates from strings
          const firstConv = {
            ...parsed[0],
            messages: parsed[0].messages.map((m) => ({
              ...m,
              timestamp: new Date(m.timestamp),
            })),
          };
          setActiveConversationId(firstConv.id);
          setMessages(firstConv.messages);
        } else {
          createNewConversation();
        }
      } catch (e) {
        console.error("Failed to load conversations:", e);
        createNewConversation();
      }
    } else {
      createNewConversation();
    }
  }, [userId]);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = "en-US";

        recognitionRef.current.onstart = () => {
          setIsListening(true);
          setError("");
        };

        recognitionRef.current.onend = () => {
          setIsListening(false);
        };

        recognitionRef.current.onresult = (event) => {
          let interimTranscript = "";

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i].transcript;

            if (event.results[i].isFinal) {
              setInput((prev) => prev + transcript);
            } else {
              interimTranscript += transcript;
            }
          }

          if (interimTranscript) {
            setInput((prev) => {
              const parts = prev.split(/\s+/);
              parts[parts.length - 1] = interimTranscript;
              return parts.join(" ");
            });
          }
        };

        recognitionRef.current.onerror = (event) => {
          console.error("Speech recognition error:", event.error);
          setError(`Microphone error: ${event.error}`);
          setIsListening(false);
        };
      }

      synthRef.current = window.speechSynthesis;
    }
  }, []);

  // Save messages when they change
  useEffect(() => {
    if (activeConversationId && messages.length > 1) {
      saveConversation();
    }
  }, [messages, activeConversationId, userId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Conversation Management Functions
  const createNewConversation = () => {
    const id = `conv-${Date.now()}`;
    const initialMessages = [
      {
        id: "1",
        role: "assistant",
        content:
          "Hello! I'm your mental health companion. How can I help you today?",
        timestamp: new Date(),
      },
    ];

    const newConversation = {
      id,
      title: "New Conversation",
      messages: initialMessages,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(id);
    setMessages(initialMessages);
    setError("");
  };

  const saveConversation = () => {
    setConversations((prevConversations) => {
      const updated = prevConversations.map((conv) =>
        conv.id === activeConversationId
          ? {
              ...conv,
              messages,
              updatedAt: new Date(),
              title: generateConversationTitle(),
            }
          : conv
      );
      try {
        localStorage.setItem(
          `chatbot-conversations-${userId}`,
          JSON.stringify(updated)
        );
      } catch (e) {
        console.error("Failed to save conversations:", e);
      }
      return updated;
    });
  };

  const generateConversationTitle = () => {
    if (messages.length > 1) {
      const firstUserMessage = messages.find((m) => m.role === "user");
      if (firstUserMessage) {
        return firstUserMessage.content.substring(0, 40) + "...";
      }
    }
    return new Date().toLocaleDateString();
  };

  const loadConversation = (conversationId) => {
    const conversation = conversations.find((c) => c.id === conversationId);
    if (conversation) {
      setActiveConversationId(conversationId);
      const messagesWithDates = conversation.messages.map((m) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      }));
      setMessages(messagesWithDates);
      setSidebarOpen(false);
    }
  };

  const deleteConversation = (conversationId, e) => {
    e.stopPropagation();
    const filtered = conversations.filter((c) => c.id !== conversationId);
    setConversations(filtered);
    try {
      localStorage.setItem(
        `chatbot-conversations-${userId}`,
        JSON.stringify(filtered)
      );
    } catch (e) {
      console.error("Failed to delete conversation:", e);
    }

    if (activeConversationId === conversationId) {
      if (filtered.length > 0) {
        loadConversation(filtered[0].id);
      } else {
        createNewConversation();
      }
    }
  };

  // Handle sending message with proper error handling
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    setError("");
    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageToSend = input;
    setInput("");
    setIsLoading(true);

    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/chatbot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          userId,
          message: messageToSend,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();

      // Handle multiple response formats
      let responseText =
        data.response ||
        data.message ||
        data.text ||
        data.reply ||
        data.content ||
        null;

      // If still no response, provide default message
      if (!responseText) {
        responseText = "I appreciate you reaching out. Could you tell me more about what's on your mind?";
        console.warn("API returned no response field, using default message");
      }

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: String(responseText).trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);
      setError(`Error: ${error.message}`);

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `I'm sorry, I encountered a technical issue: ${error.message}. Please try again or start a new conversation.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceInput = () => {
    if (!recognitionRef.current) {
      setError(
        "Speech recognition is not supported in your browser. Please use Chrome, Firefox, or Safari."
      );
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setInput("");
      recognitionRef.current.start();
    }
  };

  const speakText = (text) => {
    if (!synthRef.current) {
      setError("Text-to-speech is not supported in your browser.");
      return;
    }

    synthRef.current.cancel();
    setIsSpeaking(true);

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setIsSpeaking(false);
    };

    synthRef.current.speak(utterance);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="h-full flex bg-slate-50 dark:bg-slate-950">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-700 transform transition-all duration-300 overflow-hidden flex flex-col shadow-lg`}
      >
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <Button
            onClick={createNewConversation}
            className="w-full bg-linear-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white mb-4"
          >
            + New Chat
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-2 space-y-2">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                onClick={() => loadConversation(conv.id)}
                className={`p-3 rounded-lg cursor-pointer transition-all group ${
                  activeConversationId === conv.id
                    ? "bg-blue-100 dark:bg-blue-900/30 border-l-2 border-blue-500"
                    : "hover:bg-slate-100 dark:hover:bg-slate-800"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                      {conv.title}
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {new Date(conv.updatedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteConversation(conv.id, e)}
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-600 dark:text-red-400 transition-all"
                    title="Delete conversation"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 p-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
              title="Toggle sidebar"
            >
              {sidebarOpen ? "◀" : "▶"}
            </button>
            <div>
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <span className="text-2xl">🤖</span> Mental Health Companion
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {messages.length - 1} messages in conversation
              </p>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-linear-to-b  from-transparent via-slate-50/30 to-slate-50/50 dark:from-transparent dark:via-slate-900/30 dark:to-slate-900/50">
          {messages.length === 1 ? (
            <div className="flex items-center justify-center h-80 text-slate-500 dark:text-slate-400">
              <div className="text-center">
                <div className="text-4xl mb-4">💭</div>
                <p>Start a conversation and share what's on your mind</p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                } animate-fade-in`}
              >
                <div
                  className={`flex gap-3 max-w-md ${
                    message.role === "user" ? "flex-row-reverse" : ""
                  }`}
                >
                  {/* Avatar */}
                  <div
                    className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shadow-sm ${
                      message.role === "user"
                        ? "bg-blue-500 text-white"
                        : "bg-linear-to-br from-indigo-500 to-purple-500 text-white"
                    }`}
                  >
                    {message.role === "user" ? "U" : "AI"}
                  </div>

                  {/* Message Bubble */}
                  <div className="flex flex-col gap-1">
                    <div
                      className={`px-4 py-3 rounded-2xl shadow-sm ${
                        message.role === "user"
                          ? "bg-blue-500 text-white rounded-tr-none"
                          : "bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-600 rounded-tl-none"
                      }`}
                    >
                      <p className="text-sm leading-relaxed wrap-break-word">
                        {message.content}
                      </p>
                    </div>
                    <span className="text-xs text-slate-400 dark:text-slate-500 px-4">
                      {formatTime(message.timestamp)}
                    </span>

                    {message.role === "assistant" && (
                      <button
                        onClick={() => speakText(message.content)}
                        disabled={isSpeaking}
                        className="text-left px-3 py-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 rounded-lg transition-colors inline-flex items-center gap-1 w-fit"
                        title="Play audio"
                      >
                        <span>🔊</span> Listen
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}

          {/* Typing Animation */}
          {isLoading && (
            <div className="flex justify-start animate-fade-in">
              <div className="flex gap-3">
                <div className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold bg-linear-to-br from-indigo-500 to-purple-500 text-white shadow-sm">
                  AI
                </div>
                <div className="px-4 py-3 rounded-2xl bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-tl-none">
                  <div className="flex gap-2">
                    <div className="typing-dot"></div>
                    <div className="typing-dot animation-delay-200"></div>
                    <div className="typing-dot animation-delay-400"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {isSpeaking && (
            <div className="flex justify-center py-2">
              <div className="flex items-center gap-2 text-sm text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20 px-4 py-2 rounded-full">
                <span className="inline-block animate-pulse">🔊</span>
                <span>Playing audio...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-4 py-2 rounded-full max-w-md">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700 p-4 shadow-lg">
          {isListening && (
            <div className="mb-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-700 dark:text-red-300 flex items-center gap-2 animate-pulse">
              <span className="text-lg">🎤</span>
              <span>Listening... Click to stop</span>
            </div>
          )}

          <div className="flex gap-2">
            <Input
              placeholder={
                isListening
                  ? "Listening..."
                  : "Type a message or use voice..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              disabled={isLoading || isListening}
              className="flex-1 dark:bg-slate-800 dark:border-slate-600 dark:text-slate-100 dark:placeholder-slate-400"
            />

            {/* Voice Input Button */}
            <Button
              onClick={handleVoiceInput}
              disabled={isLoading}
              variant={isListening ? "destructive" : "outline"}
              title="Voice input"
              className="shrink-0 dark:bg-slate-800 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-700"
            >
              {isListening ? "🎙️" : "🎤"}
            </Button>

            {/* Send Button */}
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
              className="shrink-0 bg-linear-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white"
            >
              ➤
            </Button>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes typing {
          0%,
          60%,
          100% {
            opacity: 0.3;
            transform: translateY(0);
          }
          30% {
            opacity: 1;
            transform: translateY(-8px);
          }
        }

        .animate-fade-in {
          animation: fadeIn 0.3s ease-out;
        }

        .typing-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: currentColor;
          animation: typing 1.4s infinite;
          color: rgb(107, 114, 128);
        }

        .animation-delay-200 {
          animation-delay: 0.2s;
        }

        .animation-delay-400 {
          animation-delay: 0.4s;
        }

        :global(.dark) .typing-dot {
          color: rgb(148, 163, 184);
        }
      `}</style>
    </div>
  );
}
