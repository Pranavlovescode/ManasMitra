"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui_1/button";

const GAME_META = {
  first: { title: "Visual Search" },
  second: { title: "Selective Attention" },
  third_test: { title: "Sequence Memory" },
  fourth: { title: "Verbal Fluency" },
  fifth: { title: "Stroop Color Naming" },
  sixth: { title: "Cloze Word Completion" },
};

export default function GameRunnerPage() {
  const params = useParams();
  const router = useRouter();
  const slug = params?.slug;
  const [status, setStatus] = useState("");
  const [saved, setSaved] = useState(null);
  const iframeRef = useRef(null);

  const meta = GAME_META[slug] || { title: "Game" };
  const src = `/games/mentalcure/${slug}/index.html`;

  useEffect(() => {
    function onMessage(ev) {
      const data = ev?.data;
      if (!data || data.type !== "mentalcure:result") return;
      if (data.gameId !== slug) return;
      setStatus("Saving result...");
      const payload = data.payload || {};
      const body = {
        gameId: slug,
        score: Number(payload.score ?? 0),
        metrics: payload,
      };
      fetch("/api/games/results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(async (r) => {
          if (r.ok) {
            const j = await r.json();
            setSaved(j);
            setStatus("Saved!");
          } else {
            setStatus("Failed to save");
          }
        })
        .catch(() => setStatus("Failed to save"))
        .finally(() => {
          setTimeout(() => setStatus(""), 2500);
        });
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [slug]);

  return (
    <div className="container mx-auto px-6 py-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">{meta.title}</h1>
        <div className="flex items-center gap-2">
          {status && <span className="text-sm text-gray-600">{status}</span>}
          <Button
            variant="outline"
            onClick={() => router.push("/patient/games")}
          >
            Back
          </Button>
        </div>
      </div>

      {saved && (
        <Card>
          <CardHeader>
            <CardTitle>Last Result</CardTitle>
            <CardDescription>
              Saved {new Date(saved.createdAt).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-gray-700 flex gap-6">
              <div>
                Score: <span className="font-semibold">{saved.score}</span>
              </div>
              {typeof saved.metrics?.accuracy === "number" && (
                <div>
                  Accuracy:{" "}
                  <span className="font-semibold">
                    {Math.round(saved.metrics.accuracy * 100)}%
                  </span>
                </div>
              )}
              {typeof saved.metrics?.avgReactionMs === "number" && (
                <div>
                  Avg RT:{" "}
                  <span className="font-semibold">
                    {Math.round(saved.metrics.avgReactionMs)} ms
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="w-full h-[75vh] rounded-xl overflow-hidden border border-gray-200 bg-white">
        <iframe
          ref={iframeRef}
          src={src}
          className="w-full h-full"
          title={meta.title}
        />
      </div>
    </div>
  );
}
