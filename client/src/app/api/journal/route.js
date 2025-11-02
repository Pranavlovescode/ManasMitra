import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import connectDB from "@/lib/mongodb";
import Journal from "@/models/Journal";

export async function GET(request) {
  try {
    await connectDB();
    
    const authData = await auth(request);
    const { userId } = authData;
    console.log("Auth data:", authData);

    if (!authData.isAuthenticated || !userId) {
      console.log("Journal GET - Not authenticated, returning 401");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const queryUserId = request.nextUrl.searchParams.get("userId");
    
    // Ensure user can only access their own journals
    if (queryUserId && queryUserId !== userId) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const journals = await Journal.find({ userId: queryUserId || userId })
      .sort({ createdAt: -1 })
      .lean();

    return NextResponse.json(journals);
  } catch (error) {
    console.error("Journal GET error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function POST(request) {
  try {
    await connectDB();
    
    const authData = await auth(request);
    const { userId } = authData;
    console.log("POST userId:", userId);
    
    if (!authData.isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { 
      title, 
      content, 
      selectedPrompt, 
      mood, 
      inputMode,
      voiceAnalysisResult,
      skipAnalysis = false 
    } = body;

    // Validate required fields
    if (!title || !content) {
      return NextResponse.json(
        { error: "Title and content are required" },
        { status: 400 }
      );
    }

    const journalData = {
      userId,
      title: title.trim(),
      content: content.trim(),
      selectedPrompt: selectedPrompt || '',
      mood: mood || 'neutral',
      inputMode: inputMode || 'text',
      voiceAnalysisResult: voiceAnalysisResult || null,
    };

    // Perform CBT analysis if not skipped
    if (!skipAnalysis) {
      try {
        const analysisResponse = await fetch('http://127.0.0.1:8000/analyze/journal', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            title: journalData.title,
            content: journalData.content,
            mood: journalData.mood,
            prompt: selectedPrompt
          })
        });

        if (analysisResponse.ok) {
          const analysisData = await analysisResponse.json();
          
          // Transform the CBT analysis data to match our nested schema
          const contentAnalysis = analysisData.content_analysis || {};
          
          journalData.analysis = {
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
        } else {
          console.warn("CBT analysis failed, saving journal without analysis:", analysisResponse.statusText);
          // Continue without analysis if FastAPI is unavailable
        }
      } catch (analysisError) {
        console.warn("CBT analysis service unavailable, saving journal without analysis:", analysisError.message);
        // Continue without analysis if FastAPI is unavailable
      }
    }

    const newJournal = await Journal.create(journalData);

    return NextResponse.json(newJournal, { status: 201 });
  } catch (error) {
    console.error("Journal POST error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function PUT(request) {
  try {
    await connectDB();
    
    const authData = await auth(request);
    const { userId } = authData;
    
    if (!authData.isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { journalId, title, content, selectedPrompt, mood, reanalyze = false } = body;

    if (!journalId) {
      return NextResponse.json(
        { error: "Journal ID is required" },
        { status: 400 }
      );
    }

    const journal = await Journal.findOne({ _id: journalId, userId });
    if (!journal) {
      return NextResponse.json({ error: "Journal not found" }, { status: 404 });
    }

    const updateData = {};
    if (title) updateData.title = title.trim();
    if (content) updateData.content = content.trim();
    if (selectedPrompt !== undefined) updateData.selectedPrompt = selectedPrompt;
    if (mood) updateData.mood = mood;

    // Re-analyze if content changed or explicitly requested
    if (reanalyze && (title || content)) {
      try {
        const analysisResponse = await fetch('http://127.0.0.1:8000/analyze/journal', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            title: title || journal.title,
            content: content || journal.content,
            mood: mood || journal.mood,
            prompt: selectedPrompt || journal.selectedPrompt
          })
        });

        if (analysisResponse.ok) {
          const analysisData = await analysisResponse.json();
          
          // Transform the flat CBT analysis data to match our nested schema
          updateData.analysis = {
            contentAnalysis: {
              emotion: analysisData.emotion,
              emotionScore: analysisData.emotion_score,
              intent: analysisData.intent,
              intentScore: analysisData.intent_score,
              risk: analysisData.risk,
              riskScore: analysisData.risk_score,
              distortions: analysisData.distortions || [],
              distortionDetails: analysisData.distortion_details?.map(d => ({
                distortionType: d.distortion_type,
                confidence: d.confidence,
                emoji: d.emoji,
                explanation: d.explanation,
                reframingSuggestion: d.reframing_suggestion
              })) || [],
              reframes: analysisData.reframes || [],
              behavioralSuggestions: analysisData.behavioral_suggestions || [],
              clinicianNotes: analysisData.clinician_notes || [],
            },
            overallSentiment: analysisData.emotion || 'neutral',
            keyThemes: analysisData.distortions || [],
            therapeuticInsights: analysisData.reframes?.slice(0, 3) || [],
            progressIndicators: analysisData.behavioral_suggestions?.slice(0, 2) || [],
            recommendations: [...(analysisData.reframes || []), ...(analysisData.behavioral_suggestions || [])].slice(0, 4),
            analysisTimestamp: new Date(analysisData.analysis_timestamp),
          };
        }
      } catch (analysisError) {
        console.warn("CBT analysis service unavailable during update:", analysisError.message);
      }
    }

    const updatedJournal = await Journal.findByIdAndUpdate(
      journalId,
      updateData,
      { new: true, runValidators: true }
    );

    return NextResponse.json(updatedJournal);
  } catch (error) {
    console.error("Journal PUT error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function DELETE(request) {
  try {
    await connectDB();
    
    const authData = await auth(request);
    const { userId } = authData;
    
    if (!authData.isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const journalId = request.nextUrl.searchParams.get("journalId");
    
    if (!journalId) {
      return NextResponse.json(
        { error: "Journal ID is required" },
        { status: 400 }
      );
    }

    const journal = await Journal.findOne({ _id: journalId, userId });
    if (!journal) {
      return NextResponse.json({ error: "Journal not found" }, { status: 404 });
    }

    await Journal.findByIdAndDelete(journalId);

    return NextResponse.json({ message: "Journal deleted successfully" });
  } catch (error) {
    console.error("Journal DELETE error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
