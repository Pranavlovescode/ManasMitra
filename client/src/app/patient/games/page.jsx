"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui_1/button";

const GAMES = [
  { id: "first", title: "Visual Search", desc: "Find target shapes under time pressure.", emoji: "🎯" },
  { id: "second", title: "Selective Attention", desc: "Track moving targets and count accurately.", emoji: "👀" },
  { id: "third_test", title: "Sequence Memory", desc: "Memorize and recall sequences.", emoji: "🧠" },
  { id: "fourth", title: "Verbal Fluency", desc: "Generate valid words fast.", emoji: "🔤" },
  { id: "fifth", title: "Stroop Color Naming", desc: "Pick the color, not the word.", emoji: "🎨" },
  { id: "sixth", title: "Cloze Word Completion", desc: "Fill in missing letters to form words.", emoji: "✍️" },
];

export default function GamesIndexPage() {
  const router = useRouter();
  const params = useSearchParams();
  const [history, setHistory] = useState({});

  const showHistory = params?.get("history") === "1";

  useEffect(() => {
    if (!showHistory) return;
    (async () => {
      try {
        const res = await fetch(`/api/games/results`);
        if (!res.ok) return;
        const data = await res.json();
        const latest = {};
        for (const r of data) {
          if (!latest[r.gameId] || new Date(r.createdAt) > new Date(latest[r.gameId].createdAt)) {
            latest[r.gameId] = r;
          }
        }
        setHistory(latest);
      } catch {}
    })();
  }, [showHistory]);

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Cognitive Games</h1>
        <Button variant="outline" onClick={() => router.push("/patient/dashboard")}>Back to Dashboard</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {GAMES.map((g) => {
          const latest = history[g.id];
          return (
            <Card key={g.id} className="bg-white/80">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-xl">{g.emoji}</span>
                  {g.title}
                </CardTitle>
                <CardDescription>{g.desc}</CardDescription>
              </CardHeader>
              <CardContent className="flex items-center justify-between">
                <Button className="bg-indigo-600 hover:bg-indigo-700 text-white" onClick={() => router.push(`/patient/games/${g.id}`)}>Play</Button>
                {latest && (
                  <div className="text-xs text-gray-600">
                    <div>Last score: <span className="font-medium">{latest.score ?? 0}</span></div>
                    <div>When: {new Date(latest.createdAt).toLocaleString()}</div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
