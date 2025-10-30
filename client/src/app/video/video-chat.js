'use client';

import { useEffect, useState, useRef } from "react";

export default function VideoChat({ roomId, userId }) {
  const [peers, setPeers] = useState([]);
  const peerConnectionRef = useRef(null);
  const localStreamRef = useRef(null);

  useEffect(() => {
    initializePeerConnection();
  }, [roomId]);

  const initializePeerConnection = async () => {
    try {
      // Get local media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      localStreamRef.current = stream;

      // Create peer connection
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: ["stun:stun.l.google.com:19302"] },
          { urls: ["stun:stun1.l.google.com:19302"] }
        ],
      });

      // Add local stream tracks
      stream.getTracks().forEach((track) => {
        peerConnection.addTrack(track, stream);
      });

      // Handle remote stream
      peerConnection.ontrack = (event) => {
        console.log("Remote track received:", event.track.kind);
      };

      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          // Send ICE candidate to signaling server
          console.log("ICE candidate:", event.candidate);
        }
      };

      peerConnectionRef.current = peerConnection;
    } catch (error) {
      console.error("Failed to initialize peer connection:", error);
    }
  };

  return null;
}