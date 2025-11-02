"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import { MicIcon, KeyboardIcon } from "lucide-react";

export function InputModeToggle({ mode, onModeChange, disabled = false }) {
  return (
    <div className="flex items-center gap-2 p-1 bg-muted rounded-lg">
      <Button
        variant={mode === "text" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("text")}
        disabled={disabled}
        className="flex items-center gap-2"
      >
        <KeyboardIcon className="w-4 h-4" />
        Type
      </Button>
      <Button
        variant={mode === "voice" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("voice")}
        disabled={disabled}
        className="flex items-center gap-2"
      >
        <MicIcon className="w-4 h-4" />
        Voice
      </Button>
    </div>
  );
}