"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../ui/collapsible";
import { 
  ChevronDownIcon, 
  ChevronUpIcon, 
  HeartIcon, 
  BrainIcon, 
  AlertTriangleIcon,
  LightbulbIcon,
  TrendingUpIcon,
  RefreshCwIcon
} from "lucide-react";
import { useJournal } from "@/hooks/useJournal";

const CBT_PROMPTS = [
  "What thoughts are you having right now?",
  "What emotions are you experiencing?",
  "What triggered these feelings?",
  "How can you reframe this situation?",
  "What would you tell a friend in this situation?",
  "What evidence supports or challenges this thought?",
];

const MOOD_OPTIONS = [
  { value: "sad", label: "😢 Sad", color: "bg-blue-100 text-blue-800" },
  { value: "neutral", label: "😐 Neutral", color: "bg-gray-100 text-gray-800" },
  { value: "happy", label: "😊 Happy", color: "bg-green-100 text-green-800" },
  { value: "excited", label: "🤗 Excited", color: "bg-orange-100 text-orange-800" },
  { value: "loved", label: "🥰 Loved", color: "bg-pink-100 text-pink-800" },
];

const getEmotionEmoji = (emotion) => {
  const emojiMap = {
    happy: "😊", joy: "😄", sad: "😢", angry: "😠", fear: "😨", 
    anxious: "😰", calm: "😌", excited: "🤗", grateful: "🙏",
    frustrated: "😤", confused: "😕", hopeful: "🌟"
  };
  return emojiMap[emotion?.toLowerCase()] || "💭";
};

const getRiskColor = (riskScore) => {
  if (riskScore < 0.2) return "bg-green-100 text-green-800";
  if (riskScore < 0.4) return "bg-yellow-100 text-yellow-800";
  if (riskScore < 0.6) return "bg-orange-100 text-orange-800";
  return "bg-red-100 text-red-800";
};

export default function JournalModule({ userId }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [mood, setMood] = useState("neutral");
  const [selectedPrompt, setSelectedPrompt] = useState("");
  const [expandedJournal, setExpandedJournal] = useState(null);

  const {
    journals,
    isLoading,
    isAnalyzing,
    message,
    createJournal,
    analyzeJournal,
    clearMessage
  } = useJournal(userId);

  const handleSubmit = async () => {
    if (!title || !content) {
      return;
    }

    const result = await createJournal({
      title,
      content,
      selectedPrompt,
      mood,
    });

    if (result.success) {
      setTitle("");
      setContent("");
      setSelectedPrompt("");
      setMood("neutral");
    }
  };

  const handleAnalyzeJournal = async (journalId) => {
    await analyzeJournal(journalId);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Write Your Journal</CardTitle>
          <CardDescription>
            Reflect on your thoughts and feelings with guided prompts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Guided Prompts (Optional)</label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {CBT_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => setSelectedPrompt(selectedPrompt === prompt ? "" : prompt)}
                  className={`p-3 text-left text-sm rounded-lg transition-all ${
                    selectedPrompt === prompt
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Title</label>
              <Input
                placeholder="Give your entry a title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">How are you feeling?</label>
              <select
                value={mood}
                onChange={(e) => setMood(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {MOOD_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">
              Your Thoughts {selectedPrompt && `- ${selectedPrompt}`}
            </label>
            <Textarea
              placeholder="Write freely about your thoughts and feelings..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="min-h-32"
            />
          </div>

          {!title || !content ? (
            <div className="p-3 rounded-md text-sm bg-yellow-50 border border-yellow-200 text-yellow-800">
              💡 Please fill in both title and content to save your journal entry
            </div>
          ) : null}

          {message && (
            <div
              className={`p-3 rounded-md text-sm flex items-center justify-between ${
                message.includes("saved") || message.includes("completed")
                  ? "bg-green-50 border border-green-200 text-green-800"
                  : "bg-red-50 border border-red-200 text-red-800"
              }`}
            >
              <span>{message}</span>
              <button 
                onClick={clearMessage}
                className="text-xs opacity-70 hover:opacity-100"
              >
                ×
              </button>
            </div>
          )}

          <Button
            onClick={handleSubmit}
            disabled={isLoading || !title || !content}
            className="w-full"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Saving & Analyzing...
              </div>
            ) : (
              "Save Entry"
            )}
          </Button>
        </CardContent>
      </Card>

      {journals.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Your Journal Entries</CardTitle>
            <CardDescription>
              Click on entries to view AI insights and therapeutic recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {journals.slice(0, 5).map((journal) => (
                <Collapsible 
                  key={journal._id}
                  open={expandedJournal === journal._id}
                  onOpenChange={(open) => setExpandedJournal(open ? journal._id : null)}
                >
                  <div className="border rounded-lg p-4 space-y-3">
                    <CollapsibleTrigger className="w-full text-left">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold">{journal.title}</h3>
                            {journal.mood && (
                              <Badge className={MOOD_OPTIONS.find(m => m.value === journal.mood)?.color}>
                                {MOOD_OPTIONS.find(m => m.value === journal.mood)?.label}
                              </Badge>
                            )}
                            {journal.analysis?.contentAnalysis?.emotion && (
                              <Badge variant="outline">
                                {getEmotionEmoji(journal.analysis.contentAnalysis.emotion)} {journal.analysis.contentAnalysis.emotion}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-2">
                            {journal.content}
                          </p>
                          <div className="flex items-center justify-between mt-2">
                            <p className="text-xs text-muted-foreground">
                              {new Date(journal.createdAt).toLocaleDateString()}
                            </p>
                            {journal.analysis && (
                              <Badge className="bg-blue-100 text-blue-800">
                                <BrainIcon className="w-3 h-3 mr-1" />
                                AI Analyzed
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div className="ml-4">
                          {expandedJournal === journal._id ? (
                            <ChevronUpIcon className="w-4 h-4" />
                          ) : (
                            <ChevronDownIcon className="w-4 h-4" />
                          )}
                        </div>
                      </div>
                    </CollapsibleTrigger>

                    <CollapsibleContent className="space-y-4">
                      {journal.analysis ? (
                        <div className="space-y-4 mt-4 pt-4 border-t">
                          {/* Analysis Results */}
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {journal.analysis.contentAnalysis?.emotion && (
                              <div className="p-3 bg-blue-50 rounded-lg">
                                <div className="flex items-center gap-2 mb-1">
                                  <HeartIcon className="w-4 h-4 text-blue-600" />
                                  <span className="text-sm font-medium">Emotion</span>
                                </div>
                                <p className="text-sm">
                                  {getEmotionEmoji(journal.analysis.contentAnalysis.emotion)} {journal.analysis.contentAnalysis.emotion}
                                  {journal.analysis.contentAnalysis.emotionScore && (
                                    <span className="text-xs text-muted-foreground ml-1">
                                      ({Math.round(journal.analysis.contentAnalysis.emotionScore * 100)}%)
                                    </span>
                                  )}
                                </p>
                              </div>
                            )}

                            {journal.analysis.overallSentiment && (
                              <div className="p-3 bg-green-50 rounded-lg">
                                <div className="flex items-center gap-2 mb-1">
                                  <TrendingUpIcon className="w-4 h-4 text-green-600" />
                                  <span className="text-sm font-medium">Sentiment</span>
                                </div>
                                <p className="text-sm capitalize">{journal.analysis.overallSentiment}</p>
                              </div>
                            )}

                            {journal.analysis.contentAnalysis?.riskScore !== undefined && (
                              <div className="p-3 bg-orange-50 rounded-lg">
                                <div className="flex items-center gap-2 mb-1">
                                  <AlertTriangleIcon className="w-4 h-4 text-orange-600" />
                                  <span className="text-sm font-medium">Well-being</span>
                                </div>
                                <Badge className={getRiskColor(journal.analysis.contentAnalysis.riskScore)}>
                                  {journal.analysis.contentAnalysis.riskScore < 0.2 ? "Good" :
                                   journal.analysis.contentAnalysis.riskScore < 0.4 ? "Monitor" :
                                   journal.analysis.contentAnalysis.riskScore < 0.6 ? "Attention" : "Concern"}
                                </Badge>
                              </div>
                            )}
                          </div>

                          {/* Key Themes */}
                          {journal.analysis.keyThemes?.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="text-sm font-medium flex items-center gap-2">
                                <LightbulbIcon className="w-4 h-4" />
                                Key Themes
                              </h4>
                              <div className="flex flex-wrap gap-1">
                                {journal.analysis.keyThemes.map((theme, idx) => (
                                  <Badge key={idx} variant="secondary" className="text-xs">
                                    {theme}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Cognitive Distortions */}
                          {journal.analysis.contentAnalysis?.distortions?.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="text-sm font-medium flex items-center gap-2">
                                <BrainIcon className="w-4 h-4" />
                                Thought Patterns
                              </h4>
                              <div className="flex flex-wrap gap-1">
                                {journal.analysis.contentAnalysis.distortions.map((distortion, idx) => (
                                  <Badge key={idx} variant="outline" className="text-xs">
                                    {distortion.replace(/_/g, ' ')}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Therapeutic Insights */}
                          {journal.analysis.therapeuticInsights?.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="text-sm font-medium">💡 Therapeutic Insights</h4>
                              <ul className="space-y-1">
                                {journal.analysis.therapeuticInsights.map((insight, idx) => (
                                  <li key={idx} className="text-sm text-muted-foreground pl-4 border-l-2 border-blue-200">
                                    {insight}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Recommendations */}
                          {journal.analysis.recommendations?.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="text-sm font-medium">🎯 Recommendations</h4>
                              <ul className="space-y-1">
                                {journal.analysis.recommendations.map((rec, idx) => (
                                  <li key={idx} className="text-sm text-muted-foreground pl-4 border-l-2 border-green-200">
                                    {rec}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Progress Indicators */}
                          {journal.analysis.progressIndicators?.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="text-sm font-medium">📈 Progress Indicators</h4>
                              <ul className="space-y-1">
                                {journal.analysis.progressIndicators.map((indicator, idx) => (
                                  <li key={idx} className="text-sm text-green-600 pl-4 border-l-2 border-green-300">
                                    ✓ {indicator}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="mt-4 pt-4 border-t">
                          <div className="text-center py-4">
                            <p className="text-sm text-muted-foreground mb-3">
                              No analysis available for this entry
                            </p>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => handleAnalyzeJournal(journal._id)}
                              disabled={isAnalyzing}
                            >
                              <RefreshCwIcon className={`w-4 h-4 mr-2 ${isAnalyzing ? 'animate-spin' : ''}`} />
                              {isAnalyzing ? 'Analyzing...' : 'Analyze Now'}
                            </Button>
                          </div>
                        </div>
                      )}
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
