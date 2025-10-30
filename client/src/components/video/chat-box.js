"use client";

import { useEffect, useState, useRef } from "react";
import { Button } from "../ui_1/button";
import { Input } from "../ui_1/input";
import { CardContent, CardHeader, CardTitle } from "../ui_1/card";

export default function ChatBox({ roomId, userId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (!input.trim()) return;

    const newMessage = {
      id: Date.now().toString(),
      userId,
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newMessage]);
    setInput("");

    // Send message to signaling server
    console.log("Sending message:", newMessage);
  };

  return (
    <div className="flex flex-col h-full">
      <CardHeader className="border-b">
        <CardTitle className="text-lg">Session Chat</CardTitle>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <p className="text-center text-muted-foreground text-sm py-8">
            No messages yet
          </p>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.userId === userId ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
                  message.userId === userId
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                }`}
              >
                {message.content}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </CardContent>

      <div className="border-t p-3 flex gap-2">
        <Input
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
          className="text-sm"
        />
        <Button size="sm" onClick={handleSendMessage}>
          Send
        </Button>
      </div>
    </div>
  );
}
