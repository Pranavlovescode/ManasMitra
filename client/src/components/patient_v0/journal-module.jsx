// "use client";

// import { useState } from "react";
// import { Button } from "../ui/button";
// import {
//   Card,
//   CardContent,
//   CardDescription,
//   CardHeader,
//   CardTitle,
// } from "../ui/card";
// import { Input } from "../ui/input";
// import { Textarea } from "../ui/textarea";
// import { Badge } from "../ui/badge";
// import {
//   Collapsible,
//   CollapsibleContent,
//   CollapsibleTrigger,
// } from "../ui/collapsible";
// import { 
//   ChevronDownIcon, 
//   ChevronUpIcon, 
//   HeartIcon, 
//   BrainIcon, 
//   AlertTriangleIcon,
//   LightbulbIcon,
//   TrendingUpIcon,
//   RefreshCwIcon
// } from "lucide-react";
// import { useJournal } from "@/hooks/useJournal";

// const CBT_PROMPTS = [
//   "What thoughts are you having right now?",
//   "What emotions are you experiencing?",
//   "What triggered these feelings?",
//   "How can you reframe this situation?",
//   "What would you tell a friend in this situation?",
//   "What evidence supports or challenges this thought?",
// ];

// const MOOD_OPTIONS = [
//   { value: "sad", label: "😢 Sad", color: "bg-blue-100 text-blue-800" },
//   { value: "neutral", label: "😐 Neutral", color: "bg-gray-100 text-gray-800" },
//   { value: "happy", label: "😊 Happy", color: "bg-green-100 text-green-800" },
//   { value: "excited", label: "🤗 Excited", color: "bg-orange-100 text-orange-800" },
//   { value: "loved", label: "🥰 Loved", color: "bg-pink-100 text-pink-800" },
// ];

// const getEmotionEmoji = (emotion) => {
//   const emojiMap = {
//     happy: "😊", joy: "😄", sad: "😢", angry: "😠", fear: "😨", 
//     anxious: "😰", calm: "😌", excited: "🤗", grateful: "🙏",
//     frustrated: "😤", confused: "😕", hopeful: "🌟"
//   };
//   return emojiMap[emotion?.toLowerCase()] || "💭";
// };

// const getRiskColor = (riskScore) => {
//   if (riskScore < 0.2) return "bg-green-100 text-green-800";
//   if (riskScore < 0.4) return "bg-yellow-100 text-yellow-800";
//   if (riskScore < 0.6) return "bg-orange-100 text-orange-800";
//   return "bg-red-100 text-red-800";
// };

// export default function JournalModule({ userId }) {
//   const [title, setTitle] = useState("");
//   const [content, setContent] = useState("");
//   const [mood, setMood] = useState("neutral");
//   const [selectedPrompt, setSelectedPrompt] = useState("");
//   const [expandedJournal, setExpandedJournal] = useState(null);

//   const {
//     journals,
//     isLoading,
//     isAnalyzing,
//     message,
//     createJournal,
//     analyzeJournal,
//     clearMessage
//   } = useJournal(userId);
//   console.log(useJournal(userId));

//   const handleSubmit = async () => {
//     if (!title || !content) {
//       return;
//     }

//     const result = await createJournal({
//       title,
//       content,
//       selectedPrompt,
//       mood,
//     });

//     if (result.success) {
//       setTitle("");
//       setContent("");
//       setSelectedPrompt("");
//       setMood("neutral");
//     }
//   };

//   const handleAnalyzeJournal = async (journalId) => {
//     await analyzeJournal(journalId);
//   };

//   return (
//     <div className="space-y-6">
//       <Card>
//         <CardHeader>
//           <CardTitle>Write Your Journal</CardTitle>
//           <CardDescription>
//             Reflect on your thoughts and feelings with guided prompts
//           </CardDescription>
//         </CardHeader>
//         <CardContent className="space-y-4">
//           <div className="space-y-2">
//             <label className="text-sm font-medium">Guided Prompts (Optional)</label>
//             <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
//               {CBT_PROMPTS.map((prompt) => (
//                 <button
//                   key={prompt}
//                   onClick={() => setSelectedPrompt(selectedPrompt === prompt ? "" : prompt)}
//                   className={`p-3 text-left text-sm rounded-lg transition-all ${
//                     selectedPrompt === prompt
//                       ? "bg-primary text-primary-foreground"
//                       : "bg-muted hover:bg-muted/80"
//                   }`}
//                 >
//                   {prompt}
//                 </button>
//               ))}
//             </div>
//           </div>

//           <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//             <div className="space-y-2">
//               <label className="text-sm font-medium">Title</label>
//               <Input
//                 placeholder="Give your entry a title"
//                 value={title}
//                 onChange={(e) => setTitle(e.target.value)}
//               />
//             </div>

//             <div className="space-y-2">
//               <label className="text-sm font-medium">How are you feeling?</label>
//               <select
//                 value={mood}
//                 onChange={(e) => setMood(e.target.value)}
//                 className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
//               >
//                 {MOOD_OPTIONS.map((option) => (
//                   <option key={option.value} value={option.value}>
//                     {option.label}
//                   </option>
//                 ))}
//               </select>
//             </div>
//           </div>

//           <div className="space-y-2">
//             <label className="text-sm font-medium">
//               Your Thoughts {selectedPrompt && `- ${selectedPrompt}`}
//             </label>
//             <Textarea
//               placeholder="Write freely about your thoughts and feelings..."
//               value={content}
//               onChange={(e) => setContent(e.target.value)}
//               className="min-h-32"
//             />
//           </div>

//           {!title || !content ? (
//             <div className="p-3 rounded-md text-sm bg-yellow-50 border border-yellow-200 text-yellow-800">
//               💡 Please fill in both title and content to save your journal entry
//             </div>
//           ) : null}

//           {message && (
//             <div
//               className={`p-3 rounded-md text-sm flex items-center justify-between ${
//                 message.includes("saved") || message.includes("completed")
//                   ? "bg-green-50 border border-green-200 text-green-800"
//                   : "bg-red-50 border border-red-200 text-red-800"
//               }`}
//             >
//               <span>{message}</span>
//               <button 
//                 onClick={clearMessage}
//                 className="text-xs opacity-70 hover:opacity-100"
//               >
//                 ×
//               </button>
//             </div>
//           )}

//           <Button
//             onClick={handleSubmit}
//             disabled={isLoading || !title || !content}
//             className="w-full"
//           >
//             {isLoading ? (
//               <div className="flex items-center gap-2">
//                 <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
//                 Saving & Analyzing...
//               </div>
//             ) : (
//               "Save Entry"
//             )}
//           </Button>
//         </CardContent>
//       </Card>

//       {journals.length > 0 && (
//         <Card>
//           <CardHeader>
//             <CardTitle>Your Journal Entries</CardTitle>
//             <CardDescription>
//               Click on entries to view AI insights and therapeutic recommendations
//             </CardDescription>
//           </CardHeader>
//           <CardContent>
//             <div className="space-y-4">
//               {journals.slice(0, 5).map((journal) => (
//                 <Collapsible 
//                   key={journal._id}
//                   open={expandedJournal === journal._id}
//                   onOpenChange={(open) => setExpandedJournal(open ? journal._id : null)}
//                 >
//                   <div className="border rounded-lg p-4 space-y-3">
//                     <CollapsibleTrigger className="w-full text-left">
//                       <div className="flex items-center justify-between">
//                         <div className="flex-1">
//                           <div className="flex items-center gap-2 mb-2">
//                             <h3 className="font-semibold">{journal.title}</h3>
//                             {journal.mood && (
//                               <Badge className={MOOD_OPTIONS.find(m => m.value === journal.mood)?.color}>
//                                 {MOOD_OPTIONS.find(m => m.value === journal.mood)?.label}
//                               </Badge>
//                             )}
//                             {journal.analysis?.contentAnalysis?.emotion && (
//                               <Badge variant="outline">
//                                 {getEmotionEmoji(journal.analysis.emotion)} {journal.analysis.emotion}
//                               </Badge>
//                             )}
//                           </div>
//                           <p className="text-sm text-muted-foreground line-clamp-2">
//                             {journal.content}
//                           </p>
//                           <div className="flex items-center justify-between mt-2">
//                             <p className="text-xs text-muted-foreground">
//                               {new Date(journal.createdAt).toLocaleDateString()}
//                             </p>
//                             {journal.analysis && (
//                               <Badge className="bg-blue-100 text-blue-800">
//                                 <BrainIcon className="w-3 h-3 mr-1" />
//                                 AI Analyzed
//                               </Badge>
//                             )}
//                           </div>
//                         </div>
//                         <div className="ml-4">
//                           {expandedJournal === journal._id ? (
//                             <ChevronUpIcon className="w-4 h-4" />
//                           ) : (
//                             <ChevronDownIcon className="w-4 h-4" />
//                           )}
//                         </div>
//                       </div>
//                     </CollapsibleTrigger>

//                     <CollapsibleContent className="space-y-4">
//                       {journal.analysis ? (
//                         <div className="space-y-4 mt-4 pt-4 border-t">
//                           {/* Analysis Results */}
//                           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
//                             {journal.analysis?.emotion && (
//                               <div className="p-3 bg-blue-50 rounded-lg">
//                                 <div className="flex items-center gap-2 mb-1">
//                                   <HeartIcon className="w-4 h-4 text-blue-600" />
//                                   <span className="text-sm font-medium">Emotion</span>
//                                 </div>
//                                 <p className="text-sm">
//                                   {getEmotionEmoji(journal.analysis.emotion)} {journal.analysis.emotion}
//                                   {journal.analysis.emotion_score && (
//                                     <span className="text-xs text-muted-foreground ml-1">
//                                       ({Math.round(journal.analysis.emotion_score * 100)}%)
//                                     </span>
//                                   )}
//                                 </p>
//                               </div>
//                             )}

//                             {journal.analysis?.intent && (
//                               <div className="p-3 bg-green-50 rounded-lg">
//                                 <div className="flex items-center gap-2 mb-1">
//                                   <TrendingUpIcon className="w-4 h-4 text-green-600" />
//                                   <span className="text-sm font-medium">Intent</span>
//                                 </div>
//                                 <p className="text-sm capitalize">{journal.analysis.intent}</p>
//                                 {journal.analysis.intent_score && (
//                                   <span className="text-xs text-muted-foreground">
//                                     ({Math.round(journal.analysis.intent_score * 100)}%)
//                                   </span>
//                                 )}
//                               </div>
//                             )}

//                             {journal.analysis?.risk_score !== undefined && (
//                               <div className="p-3 bg-orange-50 rounded-lg">
//                                 <div className="flex items-center gap-2 mb-1">
//                                   <AlertTriangleIcon className="w-4 h-4 text-orange-600" />
//                                   <span className="text-sm font-medium">Well-being</span>
//                                 </div>
//                                 <Badge className={getRiskColor(journal.analysis.risk_score)}>
//                                   {journal.analysis.risk_score < 0.2 ? "Good" :
//                                    journal.analysis.risk_score < 0.4 ? "Monitor" :
//                                    journal.analysis.risk_score < 0.6 ? "Attention" : "Concern"}
//                                 </Badge>
//                                 <p className="text-xs text-muted-foreground mt-1">
//                                   Risk: {journal.analysis.risk || 'Unknown'}
//                                 </p>
//                               </div>
//                             )}
//                           </div>

//                           {/* Cognitive Distortions */}
//                           {journal.analysis?.distortions?.length > 0 && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium flex items-center gap-2">
//                                 <BrainIcon className="w-4 h-4" />
//                                 Thought Patterns
//                               </h4>
//                               <div className="flex flex-wrap gap-1">
//                                 {journal.analysis.distortions.map((distortion, idx) => (
//                                   <Badge key={idx} variant="outline" className="text-xs">
//                                     {distortion.replace(/_/g, ' ')}
//                                   </Badge>
//                                 ))}
//                               </div>
//                             </div>
//                           )}

//                           {/* Distortion Details */}
//                           {journal.analysis?.distortion_details?.length > 0 && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium flex items-center gap-2">
//                                 <LightbulbIcon className="w-4 h-4" />
//                                 Distortion Details
//                               </h4>
//                               <div className="space-y-2">
//                                 {journal.analysis.distortion_details.map((detail, idx) => (
//                                   <div key={idx} className="p-2 bg-yellow-50 rounded border-l-4 border-yellow-400">
//                                     <div className="flex items-center gap-2 mb-1">
//                                       <span>{detail.emoji}</span>
//                                       <span className="text-sm font-medium">
//                                         {detail.distortion_type.replace(/_/g, ' ')}
//                                       </span>
//                                       <Badge variant="secondary" className="text-xs">
//                                         {Math.round(detail.confidence * 100)}%
//                                       </Badge>
//                                     </div>
//                                     {detail.explanation && (
//                                       <p className="text-xs text-muted-foreground">{detail.explanation}</p>
//                                     )}
//                                     {detail.reframing_suggestion && (
//                                       <p className="text-xs text-green-700 mt-1">💡 {detail.reframing_suggestion}</p>
//                                     )}
//                                   </div>
//                                 ))}
//                               </div>
//                             </div>
//                           )}

//                           {/* Reframes */}
//                           {journal.analysis?.reframes?.length > 0 && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium">� Reframing Suggestions</h4>
//                               <ul className="space-y-1">
//                                 {journal.analysis.reframes.map((reframe, idx) => (
//                                   <li key={idx} className="text-sm text-blue-700 pl-4 border-l-2 border-blue-200">
//                                     {reframe}
//                                   </li>
//                                 ))}
//                               </ul>
//                             </div>
//                           )}

//                           {/* Behavioral Suggestions */}
//                           {journal.analysis?.behavioral_suggestions?.length > 0 && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium">🎯 Behavioral Suggestions</h4>
//                               <ul className="space-y-1">
//                                 {journal.analysis.behavioral_suggestions.map((suggestion, idx) => (
//                                   <li key={idx} className="text-sm text-green-700 pl-4 border-l-2 border-green-200">
//                                     {suggestion}
//                                   </li>
//                                 ))}
//                               </ul>
//                             </div>
//                           )}

//                           {/* Clinician Notes */}
//                           {journal.analysis?.clinician_notes?.length > 0 && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium">� Clinical Notes</h4>
//                               <ul className="space-y-1">
//                                 {journal.analysis.clinician_notes.map((note, idx) => (
//                                   <li key={idx} className="text-sm text-purple-700 pl-4 border-l-2 border-purple-200">
//                                     {note}
//                                   </li>
//                                 ))}
//                               </ul>
//                             </div>
//                           )}

//                           {/* User Facing Summary */}
//                           {journal.analysis?.user_facing && (
//                             <div className="space-y-2">
//                               <h4 className="text-sm font-medium">💬 Summary</h4>
//                               <div className="p-3 bg-gray-50 rounded-lg border-l-4 border-gray-400">
//                                 <p className="text-sm text-gray-700">{journal.analysis.user_facing}</p>
//                               </div>
//                             </div>
//                           )}

//                           {/* Analysis Metadata */}
//                           {journal.analysis?.analysis_timestamp && (
//                             <div className="pt-2 border-t border-gray-200">
//                               <p className="text-xs text-muted-foreground">
//                                 Analysis completed: {new Date(journal.analysis.analysis_timestamp).toLocaleString()}
//                               </p>
//                               {journal.analysis.emotion_source && (
//                                 <p className="text-xs text-muted-foreground">
//                                   Emotion source: {journal.analysis.emotion_source}
//                                 </p>
//                               )}
//                             </div>
//                           )}
//                         </div>
//                       ) : (
//                         <div className="mt-4 pt-4 border-t">
//                           <div className="text-center py-4">
//                             <p className="text-sm text-muted-foreground mb-3">
//                               No analysis available for this entry
//                             </p>
//                             <Button 
//                               variant="outline" 
//                               size="sm"
//                               onClick={() => handleAnalyzeJournal(journal._id)}
//                               disabled={isAnalyzing}
//                             >
//                               <RefreshCwIcon className={`w-4 h-4 mr-2 ${isAnalyzing ? 'animate-spin' : ''}`} />
//                               {isAnalyzing ? 'Analyzing...' : 'Analyze Now'}
//                             </Button>
//                           </div>
//                         </div>
//                       )}
//                     </CollapsibleContent>
//                   </div>
//                 </Collapsible>
//               ))}
//             </div>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   );
// }
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
  RefreshCwIcon,
} from "lucide-react";
import { useJournal } from "@/hooks/useJournal";
import { VoiceRecorder } from "../voice/voice-recorder";
import { InputModeToggle } from "../voice/input-mode-toggle";

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
    happy: "😊",
    joy: "😄",
    sad: "😢",
    angry: "😠",
    fear: "😨",
    anxious: "😰",
    calm: "😌",
    excited: "🤗",
    grateful: "🙏",
    frustrated: "😤",
    confused: "😕",
    hopeful: "🌟",
    optimism: "🌞",
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
  const [inputMode, setInputMode] = useState("text");
  const [voiceAnalysisResult, setVoiceAnalysisResult] = useState(null);
  const [isVoiceProcessing, setIsVoiceProcessing] = useState(false);

  const {
    journals,
    isLoading,
    isAnalyzing,
    message,
    createJournal,
    analyzeJournal,
    clearMessage,
  } = useJournal(userId);

  const handleSubmit = async () => {
    if (!title || !content) return;

    const journalData = {
      title,
      content,
      selectedPrompt,
      mood,
      inputMode,
      voiceAnalysisResult: inputMode === "voice" ? voiceAnalysisResult : null,
    };

    const result = await createJournal(journalData);

    if (result.success) {
      setTitle("");
      setContent("");
      setSelectedPrompt("");
      setMood("neutral");
      setInputMode("text");
      setVoiceAnalysisResult(null);
      setIsVoiceProcessing(false);
    }
  };

  const handleAnalyzeJournal = async (journalId) => {
    await analyzeJournal(journalId);
  };

  const handleVoiceRecordingComplete = (blob) => {
    // Audio blob is ready for analysis
    console.log("Voice recording completed, size:", blob.size);
    setIsVoiceProcessing(true);
  };

  const handleVoiceAnalysis = (analysisResult) => {
    setIsVoiceProcessing(false);
    setVoiceAnalysisResult(analysisResult);
    
    if (analysisResult.transcription) {
      // Auto-fill content with transcription
      setContent(analysisResult.transcription);
      
      // If no title is set, generate one from the first part of transcription
      if (!title && analysisResult.transcription) {
        const firstSentence = analysisResult.transcription.split('.')[0].trim();
        const autoTitle = firstSentence.length > 50 
          ? firstSentence.substring(0, 47) + "..." 
          : firstSentence;
        setTitle(autoTitle || "Voice Journal Entry");
      }
      
      // Auto-detect mood from voice emotion if available
      if (analysisResult.emotion && analysisResult.emotion_source === 'voice') {
        const voiceEmotion = analysisResult.emotion.toLowerCase();
        // Map voice emotions to our mood options
        const emotionMoodMap = {
          'happy': 'happy',
          'joy': 'happy', 
          'excited': 'excited',
          'sad': 'sad',
          'neutral': 'neutral',
          'calm': 'neutral'
        };
        const detectedMood = emotionMoodMap[voiceEmotion] || mood;
        setMood(detectedMood);
      }
    }
  };

  const handleInputModeChange = (mode) => {
    setInputMode(mode);
    setVoiceAnalysisResult(null);
    if (mode === "text") {
      // Clear voice-related state when switching to text mode
      setIsVoiceProcessing(false);
    }
  };

  const clearVoiceInput = () => {
    setVoiceAnalysisResult(null);
    setContent("");
    setTitle("");
    setMood("neutral");
  };

  return (
    <div className="space-y-6">
      {/* --- Journal Creation Card --- */}
      <Card>
        <CardHeader>
          <CardTitle>Write Your Journal</CardTitle>
          <CardDescription>
            Reflect on your thoughts and feelings with guided prompts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Input Mode Toggle */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Input Method</label>
              <InputModeToggle 
                mode={inputMode} 
                onModeChange={handleInputModeChange} 
                disabled={isLoading || isVoiceProcessing}
              />
            </div>
            
            {inputMode === "voice" && (
              <div className="space-y-3">
                <VoiceRecorder 
                  onRecordingComplete={handleVoiceRecordingComplete}
                  onTranscription={handleVoiceAnalysis}
                  disabled={isLoading}
                />
                
                {voiceAnalysisResult && (
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium text-blue-800">Voice Analysis Results</h4>
                      <Button variant="ghost" size="sm" onClick={clearVoiceInput}>
                        Clear
                      </Button>
                    </div>
                    
                    {voiceAnalysisResult.transcription && (
                      <div>
                        <p className="text-xs text-blue-600 font-medium">Transcription:</p>
                        <p className="text-sm text-blue-800">{voiceAnalysisResult.transcription}</p>
                      </div>
                    )}
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {voiceAnalysisResult.emotion && (
                        <div className="text-xs">
                          <span className="font-medium text-blue-600">Emotion:</span>
                          <p className="text-blue-800">{voiceAnalysisResult.emotion}</p>
                        </div>
                      )}
                      
                      {voiceAnalysisResult.intent && (
                        <div className="text-xs">
                          <span className="font-medium text-blue-600">Intent:</span>
                          <p className="text-blue-800">{voiceAnalysisResult.intent}</p>
                        </div>
                      )}
                      
                      {voiceAnalysisResult.risk && (
                        <div className="text-xs">
                          <span className="font-medium text-blue-600">Well-being:</span>
                          <Badge className={getRiskColor(voiceAnalysisResult.risk_score || 0)}>
                            {voiceAnalysisResult.risk}
                          </Badge>
                        </div>
                      )}
                      
                      {voiceAnalysisResult.emotion_source && (
                        <div className="text-xs">
                          <span className="font-medium text-blue-600">Source:</span>
                          <p className="text-blue-800">{voiceAnalysisResult.emotion_source}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Prompts */}
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

          {/* Title + Mood */}
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
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                {MOOD_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Content */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                Your Thoughts {selectedPrompt && `- ${selectedPrompt}`}
              </label>
              {inputMode === "voice" && voiceAnalysisResult?.transcription && (
                <Badge variant="secondary" className="text-xs">
                  🎤 Voice Input
                </Badge>
              )}
            </div>
            <Textarea
              placeholder={
                inputMode === "voice" 
                  ? "Your voice transcription will appear here..."
                  : "Write freely about your thoughts and feelings..."
              }
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="min-h-32"
              disabled={inputMode === "voice" && isVoiceProcessing}
            />
            {inputMode === "voice" && !voiceAnalysisResult && (
              <p className="text-xs text-muted-foreground">
                🎙️ Use the voice recorder above to speak your journal entry. It will be transcribed and analyzed automatically.
              </p>
            )}
          </div>

          {!title || !content ? (
            <div className="p-3 rounded-md text-sm bg-yellow-50 border border-yellow-200 text-yellow-800">
              💡 Please fill in both title and content to save your journal entry
              {inputMode === "voice" && (
                <span className="block mt-1">
                  🎤 Record your voice and it will automatically fill in the content
                </span>
              )}
            </div>
          ) : null}

          {isVoiceProcessing && (
            <div className="p-3 rounded-md text-sm bg-blue-50 border border-blue-200 text-blue-800 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              Processing your voice recording...
            </div>
          )}

          {message && (
            <div
              className={`p-3 rounded-md text-sm flex items-center justify-between ${
                message.includes("saved") || message.includes("completed")
                  ? "bg-green-50 border border-green-200 text-green-800"
                  : "bg-red-50 border border-red-200 text-red-800"
              }`}
            >
              <span>{message}</span>
              <button onClick={clearMessage} className="text-xs opacity-70 hover:opacity-100">
                ×
              </button>
            </div>
          )}

          <Button onClick={handleSubmit} disabled={isLoading || !title || !content} className="w-full">
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

      {/* --- Journals List --- */}
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
              {journals.slice(0, 5).map((journal) => {
                const analysis = journal.analysis?.contentAnalysis;
                return (
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
                              {analysis?.emotion && (
                                <Badge variant="outline">
                                  {getEmotionEmoji(analysis.emotion)} {analysis.emotion}
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
                              {analysis && (
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
                        {analysis ? (
                          <div className="space-y-4 mt-4 pt-4 border-t">
                            {/* --- Analysis Results --- */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                              {/* Emotion */}
                              {analysis.emotion && (
                                <div className="p-3 bg-blue-50 rounded-lg">
                                  <div className="flex items-center gap-2 mb-1">
                                    <HeartIcon className="w-4 h-4 text-blue-600" />
                                    <span className="text-sm font-medium">Emotion</span>
                                  </div>
                                  <p className="text-sm">
                                    {getEmotionEmoji(analysis.emotion)} {analysis.emotion}
                                    {analysis.emotionScore && (
                                      <span className="text-xs text-muted-foreground ml-1">
                                        ({Math.round(analysis.emotionScore * 100)}%)
                                      </span>
                                    )}
                                  </p>
                                </div>
                              )}

                              {/* Intent */}
                              {analysis.intent && (
                                <div className="p-3 bg-green-50 rounded-lg">
                                  <div className="flex items-center gap-2 mb-1">
                                    <TrendingUpIcon className="w-4 h-4 text-green-600" />
                                    <span className="text-sm font-medium">Intent</span>
                                  </div>
                                  <p className="text-sm capitalize">{analysis.intent}</p>
                                  {analysis.intentScore && (
                                    <span className="text-xs text-muted-foreground">
                                      ({Math.round(analysis.intentScore * 100)}%)
                                    </span>
                                  )}
                                </div>
                              )}

                              {/* Risk */}
                              {analysis.riskScore !== undefined && (
                                <div className="p-3 bg-orange-50 rounded-lg">
                                  <div className="flex items-center gap-2 mb-1">
                                    <AlertTriangleIcon className="w-4 h-4 text-orange-600" />
                                    <span className="text-sm font-medium">Well-being</span>
                                  </div>
                                  <Badge className={getRiskColor(analysis.riskScore)}>
                                    {analysis.riskScore < 0.2
                                      ? "Good"
                                      : analysis.riskScore < 0.4
                                      ? "Monitor"
                                      : analysis.riskScore < 0.6
                                      ? "Attention"
                                      : "Concern"}
                                  </Badge>
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Risk: {analysis.risk || "Unknown"}
                                  </p>
                                </div>
                              )}
                            </div>

                            {/* Cognitive Distortions */}
                            {analysis.distortions?.length > 0 && (
                              <div className="space-y-2">
                                <h4 className="text-sm font-medium flex items-center gap-2">
                                  <BrainIcon className="w-4 h-4" />
                                  🧠 Cognitive Patterns Detected
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                  {analysis.distortions.map((distortion, idx) => (
                                    <Badge key={idx} variant="secondary">
                                      {distortion.replace(/_/g, ' ')}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Reframes */}
                            {analysis.reframes?.length > 0 && (
                              <div className="space-y-2">
                                <h4 className="text-sm font-medium">💡 Reframing Suggestions</h4>
                                <ul className="space-y-1">
                                  {analysis.reframes.map((reframe, idx) => (
                                    <li key={idx} className="text-sm text-blue-700 pl-4 border-l-2 border-blue-200">
                                      {reframe}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Behavioral Suggestions */}
                            {analysis.behavioralSuggestions?.length > 0 && (
                              <div className="space-y-2">
                                <h4 className="text-sm font-medium">🎯 Behavioral Suggestions</h4>
                                <ul className="space-y-1">
                                  {analysis.behavioralSuggestions.map((suggestion, idx) => (
                                    <li key={idx} className="text-sm text-green-700 pl-4 border-l-2 border-green-200">
                                      {suggestion}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Clinician Notes */}
                            {analysis.clinicianNotes?.length > 0 && (
                              <div className="space-y-2">
                                <h4 className="text-sm font-medium">🩺 Clinical Notes</h4>
                                <ul className="space-y-1">
                                  {analysis.clinicianNotes.map((note, idx) => (
                                    <li key={idx} className="text-sm text-purple-700 pl-4 border-l-2 border-purple-200">
                                      {note}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Timestamp */}
                            {analysis.analysisTimestamp && (
                              <div className="pt-2 border-t border-gray-200">
                                <p className="text-xs text-muted-foreground">
                                  Analysis completed:{" "}
                                  {new Date(analysis.analysisTimestamp).toLocaleString()}
                                </p>
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
                                <RefreshCwIcon
                                  className={`w-4 h-4 mr-2 ${isAnalyzing ? "animate-spin" : ""}`}
                                />
                                {isAnalyzing ? "Analyzing..." : "Analyze Now"}
                              </Button>
                            </div>
                          </div>
                        )}
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
