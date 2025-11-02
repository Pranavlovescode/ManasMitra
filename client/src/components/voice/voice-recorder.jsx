"use client";

import { useState, useRef, useCallback } from "react";
import { Button } from "../ui/button";
import {
  MicIcon,
  MicOffIcon,
  PlayIcon,
  PauseIcon,
  StopCircleIcon,
} from "lucide-react";

export function VoiceRecorder({
  onRecordingComplete,
  onTranscription,
  disabled = false,
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      const chunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setRecordedBlob(blob);
        onRecordingComplete?.(blob);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingDuration(0);

      // Start duration timer
      intervalRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Unable to access microphone. Please check your permissions.");
    }
  }, [onRecordingComplete]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      // Stop and clean up stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      // Clear interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [isRecording]);

  const playRecording = useCallback(() => {
    if (recordedBlob && audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        const url = URL.createObjectURL(recordedBlob);
        audioRef.current.src = url;
        audioRef.current.play();
        setIsPlaying(true);

        audioRef.current.onended = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(url);
        };
      }
    }
  }, [recordedBlob, isPlaying]);

  const analyzeRecording = useCallback(async () => {
    if (!recordedBlob) return;

    setIsProcessing(true);
    try {
      const formData = new FormData();

      // Convert webm to wav for better compatibility
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const arrayBuffer = await recordedBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Convert to WAV
      const wavBlob = audioBufferToWav(audioBuffer);
      formData.append("file", wavBlob, "recording.wav");

      const apiUrl =
        process.env.NEXT_PUBLIC_CBT_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/audio/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      onTranscription?.(result);
    } catch (error) {
      console.error("Error analyzing recording:", error);
      alert("Failed to analyze recording. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  }, [recordedBlob, onTranscription]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const clearRecording = useCallback(() => {
    setRecordedBlob(null);
    setRecordingDuration(0);
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.src = "";
    }
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        {!isRecording ? (
          <Button
            onClick={startRecording}
            disabled={disabled}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <MicIcon className="w-4 h-4" />
            Start Recording
          </Button>
        ) : (
          <Button
            onClick={stopRecording}
            variant="destructive"
            size="sm"
            className="flex items-center gap-2 animate-pulse"
          >
            <MicOffIcon className="w-4 h-4" />
            Stop Recording ({formatDuration(recordingDuration)})
          </Button>
        )}

        {recordedBlob && !isRecording && (
          <>
            <Button
              onClick={playRecording}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              {isPlaying ? (
                <>
                  <PauseIcon className="w-4 h-4" />
                  Pause
                </>
              ) : (
                <>
                  <PlayIcon className="w-4 h-4" />
                  Play ({formatDuration(recordingDuration)})
                </>
              )}
            </Button>

            <Button
              onClick={analyzeRecording}
              disabled={isProcessing}
              className="flex items-center gap-2"
            >
              {isProcessing ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                "🧠"
              )}
              {isProcessing ? "Analyzing..." : "Analyze Voice"}
            </Button>

            <Button
              onClick={clearRecording}
              variant="ghost"
              size="sm"
              className="flex items-center gap-2"
            >
              <StopCircleIcon className="w-4 h-4" />
              Clear
            </Button>
          </>
        )}
      </div>

      <audio ref={audioRef} className="hidden" />

      {isRecording && (
        <div className="flex items-center gap-2 text-sm text-red-600">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          Recording... Speak clearly into your microphone
        </div>
      )}
    </div>
  );
}

// Helper function to convert AudioBuffer to WAV
function audioBufferToWav(buffer) {
  const length = buffer.length;
  const arrayBuffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(arrayBuffer);

  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, length * 2, true);

  // Convert float samples to 16-bit PCM
  const channelData = buffer.getChannelData(0);
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}
