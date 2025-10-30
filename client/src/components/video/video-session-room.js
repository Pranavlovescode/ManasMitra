"use client";

import { useEffect, useState, useRef } from "react";
import { Button } from "../ui_1/button";
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  Share2,
  MessageCircle,
  Phone,
} from "lucide-react";
import ChatBox from "./chat-box";

export default function VideoSessionRoom({ roomId, userId, userRole }) {
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [sessionActive, setSessionActive] = useState(true);
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);

  useEffect(() => {
    initializeWebRTC();
  }, [roomId, userId]);

  const initializeWebRTC = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      // Connect to signaling server
      connectToSignalingServer();
    } catch (error) {
      console.error("Failed to get media stream:", error);
    }
  };

  const connectToSignalingServer = () => {
    // Mock WebSocket connection - In production, use actual Socket.IO
    console.log("Connecting to signaling server for room:", roomId);
  };

  const handleToggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleToggleVideo = () => {
    setIsVideoOn(!isVideoOn);
  };

  const handleScreenShare = async () => {
    try {
      if (!isScreenSharing) {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
        });
        setIsScreenSharing(true);
      } else {
        setIsScreenSharing(false);
      }
    } catch (error) {
      console.error("Screen sharing failed:", error);
    }
  };

  const handleEndCall = () => {
    setSessionActive(false);
    // Clean up resources
    if (localVideoRef.current?.srcObject) {
      const tracks = localVideoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
    }
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Main Video Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="border-b bg-white p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                Therapy Session
              </h1>
              <p className="text-sm text-gray-500">Room: {roomId}</p>
            </div>
            <div className="text-sm text-gray-500">
              {sessionActive ? "Session Active" : "Session Ended"}
            </div>
          </div>
        </header>

        {/* Video Container */}
        <div className="flex-1 flex gap-4 p-4 overflow-hidden">
          {/* Remote Video */}
          <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden relative">
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-4 left-4 text-white text-sm bg-black/50 px-3 py-1 rounded">
              Remote Participant
            </div>
          </div>

          {/* Local Video */}
          <div className="w-64 bg-gray-900 rounded-lg overflow-hidden relative">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-4 left-4 text-white text-sm bg-black/50 px-3 py-1 rounded">
              You
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="border-t bg-white p-4">
          <div className="flex items-center justify-center gap-4">
            <Button
              size="lg"
              variant={isMuted ? "destructive" : "default"}
              onClick={handleToggleMute}
              className="rounded-full w-14 h-14 p-0"
            >
              {isMuted ? (
                <MicOff className="w-6 h-6" />
              ) : (
                <Mic className="w-6 h-6" />
              )}
            </Button>

            <Button
              size="lg"
              variant={!isVideoOn ? "destructive" : "default"}
              onClick={handleToggleVideo}
              className="rounded-full w-14 h-14 p-0"
            >
              {isVideoOn ? (
                <Video className="w-6 h-6" />
              ) : (
                <VideoOff className="w-6 h-6" />
              )}
            </Button>

            <Button
              size="lg"
              variant={isScreenSharing ? "secondary" : "default"}
              onClick={handleScreenShare}
              className="rounded-full w-14 h-14 p-0"
            >
              <Share2 className="w-6 h-6" />
            </Button>

            <Button
              size="lg"
              variant="default"
              onClick={() => setShowChat(!showChat)}
              className="rounded-full w-14 h-14 p-0"
            >
              <MessageCircle className="w-6 h-6" />
            </Button>

            <Button
              size="lg"
              variant="destructive"
              onClick={handleEndCall}
              className="rounded-full w-14 h-14 p-0"
            >
              <Phone className="w-6 h-6" />
            </Button>
          </div>
        </div>
      </div>

      {/* Chat Sidebar */}
      {showChat && (
        <div className="w-80 border-l bg-white flex flex-col">
          <ChatBox roomId={roomId} userId={userId} />
        </div>
      )}
    </div>
  );
}
