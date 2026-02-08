"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui_1/button";

const GAMES = [
  {
    id: "first",
    title: "Visual Search",
    desc: "Find target shapes under time pressure.",
    emoji: "🎯",
  },
  {
    id: "second",
    title: "Selective Attention",
    desc: "Track moving targets and count accurately.",
    emoji: "👀",
  },
  {
    id: "third_test",
    title: "Sequence Memory",
    desc: "Memorize and recall sequences.",
    emoji: "🧠",
  },
  {
    id: "fourth",
    title: "Verbal Fluency",
    desc: "Generate valid words fast.",
    emoji: "🔤",
  },
  {
    id: "fifth",
    title: "Stroop Color Naming",
    desc: "Pick the color, not the word.",
    emoji: "🎨",
  },
  {
    id: "sixth",
    title: "Cloze Word Completion",
    desc: "Fill in missing letters to form words.",
    emoji: "✍️",
  },
];

export default function GamesModule({ showHistory = false, onPlayGame }) {
  const router = useRouter();
  const [history, setHistory] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!showHistory) return;
    setLoading(true);
    (async () => {
      try {
        const res = await fetch(`/api/games/results`);
        if (!res.ok) return;
        const data = await res.json();
        const latest = {};
        for (const r of data) {
          if (
            !latest[r.gameId] ||
            new Date(r.createdAt) > new Date(latest[r.gameId].createdAt)
          ) {
            latest[r.gameId] = r;
          }
        }
        setHistory(latest);
      } catch (err) {
        console.error("Error fetching game history:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [showHistory]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Cognitive Games
        </h2>
        <p className="text-gray-600">
          Practice and measure perception, attention, memory, language and
          reflexes. Your results are shared with your therapist to personalize
          care.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {GAMES.map((g) => {
          const latest = history[g.id];
          return (
            <Card
              key={g.id}
              className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">{g.emoji}</span>
                  <span className="text-lg">{g.title}</span>
                </CardTitle>
                <CardDescription className="text-sm">
                  {g.desc}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  className="w-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-md"
                  onClick={() => {
                    if (onPlayGame) {
                      onPlayGame({ id: g.id, title: g.title });
                    } else {
                      router.push(`/patient/games/${g.id}`);
                    }
                  }}
                >
                  <span className="mr-2">🎮</span>
                  Play Now
                </Button>
                {showHistory && latest && (
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Last Score:</span>
                        <span className="text-indigo-600 font-bold text-sm">
                          {latest.score ?? 0}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Date:</span>
                        <span className="text-xs">
                          {new Date(latest.createdAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                {showHistory && !latest && !loading && (
                  <div className="p-3 bg-gray-50 rounded-lg text-center">
                    <p className="text-xs text-gray-500">No scores yet</p>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {showHistory && loading && (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="text-sm text-gray-600 mt-2">Loading scores...</p>
        </div>
      )}
    </div>
  );
}
