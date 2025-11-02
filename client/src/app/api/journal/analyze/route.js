import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import connectDB from "@/lib/mongodb";
import Journal from "@/models/Journal";

export async function POST(request) {
  try {
    await connectDB();
    
    const authData = await auth(request);
    const { userId } = authData;
    
    if (!authData.isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { journalId } = body;

    if (!journalId) {
      return NextResponse.json(
        { error: "Journal ID is required" },
        { status: 400 }
      );
    }

    // Find the journal
    const journal = await Journal.findOne({ _id: journalId, userId });
    if (!journal) {
      return NextResponse.json({ error: "Journal not found" }, { status: 404 });
    }

    // Perform CBT analysis
    try {
      const analysisResponse = await fetch('http://127.0.0.1:8000/analyze/journal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: journal.title,
          content: journal.content,
          mood: journal.mood,
          prompt: journal.selectedPrompt
        })
      });

      if (!analysisResponse.ok) {
        return NextResponse.json(
          { error: "Analysis service unavailable" },
          { status: 503 }
        );
      }

      const analysisData = await analysisResponse.json();
      
      // Transform the CBT analysis data to match our nested schema
      const contentAnalysis = analysisData.content_analysis || {};
      
      const analysis = {
        contentAnalysis: {
          emotion: contentAnalysis.emotion,
          emotionScore: contentAnalysis.emotion_score,
          intent: contentAnalysis.intent,
          intentScore: contentAnalysis.intent_score,
          risk: contentAnalysis.risk,
          riskScore: contentAnalysis.risk_score,
          distortions: contentAnalysis.distortions || [],
          distortionDetails: contentAnalysis.distortion_details?.map(d => ({
            distortionType: d.distortion_type,
            confidence: d.confidence,
            emoji: d.emoji,
            explanation: d.explanation,
            reframingSuggestion: d.reframing_suggestion
          })) || [],
          reframes: contentAnalysis.reframes || [],
          behavioralSuggestions: contentAnalysis.behavioral_suggestions || [],
          clinicianNotes: contentAnalysis.clinician_notes || [],
        },
        overallSentiment: analysisData.overall_sentiment || 'neutral',
        keyThemes: analysisData.key_themes || [],
        therapeuticInsights: analysisData.therapeutic_insights || [],
        progressIndicators: analysisData.progress_indicators || [],
        recommendations: analysisData.recommendations || [],
        analysisTimestamp: new Date(analysisData.analysis_timestamp),
      };

      // Update the journal with the analysis
      const updatedJournal = await Journal.findByIdAndUpdate(
        journalId,
        { analysis },
        { new: true, runValidators: true }
      );

      return NextResponse.json({
        message: "Analysis completed successfully",
        journal: updatedJournal,
        analysis: analysisData
      });

    } catch (analysisError) {
      console.error("CBT analysis failed:", analysisError);
      return NextResponse.json(
        { error: "Analysis failed", details: analysisError.message },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error("Journal analysis error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}