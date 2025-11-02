import mongoose from "mongoose";

const JournalSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
      index: true,
    },
    title: {
      type: String,
      required: true,
      trim: true,
    },
    content: {
      type: String,
      required: true,
    },
    selectedPrompt: {
      type: String,
      default: "",
    },
    mood: {
      type: String,
      enum: ["sad", "neutral", "happy", "excited", "loved"],
      default: "neutral",
    },
    inputMode: {
      type: String,
      enum: ["text", "voice"],
      default: "text",
    },
    voiceAnalysisResult: {
      transcription: String,
      emotion: String,
      emotion_score: Number,
      emotion_source: String,
      intent: String,
      intent_score: Number,
      risk: String,
      risk_score: Number,
      escalation: Boolean,
      distortions: [String],
      distortion_details: [{
        distortion_type: String,
        confidence: Number,
        emoji: String,
        explanation: String,
        reframing_suggestion: String,
      }],
      reframes: [String],
      behavioral_suggestions: [String],
      clinician_notes: [String],
      user_facing: String,
      analysis_timestamp: Date,
      raw_analysis: mongoose.Schema.Types.Mixed,
    },
    tags: [String],
    isPrivate: {
      type: Boolean,
      default: true,
    },
    // CBT Analysis Results
    analysis: {
      contentAnalysis: {
        emotion: String,
        emotionScore: Number,
        intent: String,
        intentScore: Number,
        risk: String,
        riskScore: Number,
        distortions: [String],
        distortionDetails: [
          {
            distortionType: String,
            confidence: Number,
            emoji: String,
            explanation: String,
            reframingSuggestion: String,
          },
        ],
        reframes: [String],
        behavioralSuggestions: [String],
        clinicianNotes: [String],
      },
      titleAnalysis: {
        emotion: String,
        emotionScore: Number,
        distortions: [String],
      },
      overallSentiment: String,
      keyThemes: [String],
      therapeuticInsights: [String],
      progressIndicators: [String],
      recommendations: [String],
      analysisTimestamp: Date,
    },
  },
  {
    timestamps: true,
  }
);

// Create indexes for better performance
JournalSchema.index({ userId: 1, createdAt: -1 });
JournalSchema.index({ userId: 1, title: "text", content: "text" });

export default mongoose.models.Journal ||
  mongoose.model("Journal", JournalSchema);
