'use client';

import { useAuth } from "@clerk/nextjs";
import VideoSessionRoom from "../video/video-session-room";

export default function VideoSessionPage() {
  const { userId } = useAuth();
  
  // In a real application, you would get the roomId from your backend or URL params
  const roomId = "demo-room-1";

  return (
    <VideoSessionRoom 
      roomId={roomId} 
      userId={userId} 
      userRole="patient"
    />
  );
}