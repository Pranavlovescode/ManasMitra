import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import connectDB from "@/lib/mongodb";
import Journal from "@/models/Journal";

export async function GET(request) {
  try {
    await connectDB();
    
    const { userId , isAuthenticated} = auth(request);
    const authData = await auth(request);
    console.log("Auth data:", authData);

    if (!isAuthenticated) {
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
    
    const { userId,isAuthenticated } = await auth(request);
    console.log(userId);
    if (!isAuthenticated) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { title, content, selectedPrompt, mood, skipAnalysis = false } = body;

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
          
          // Transform the analysis data to match our schema
          journalData.analysis = {
            contentAnalysis: {
              emotion: analysisData.content_analysis?.emotion,
              emotionScore: analysisData.content_analysis?.emotion_score,
              intent: analysisData.content_analysis?.intent,
              intentScore: analysisData.content_analysis?.intent_score,
              risk: analysisData.content_analysis?.risk,
              riskScore: analysisData.content_analysis?.risk_score,
              distortions: analysisData.content_analysis?.distortions || [],
              distortionDetails: analysisData.content_analysis?.distortion_details?.map(d => ({
                distortionType: d.distortion_type,
                confidence: d.confidence,
                emoji: d.emoji,
                explanation: d.explanation,
                reframingSuggestion: d.reframing_suggestion
              })) || [],
              reframes: analysisData.content_analysis?.reframes || [],
              behavioralSuggestions: analysisData.content_analysis?.behavioral_suggestions || [],
              clinicianNotes: analysisData.content_analysis?.clinician_notes || [],
            },
            titleAnalysis: analysisData.title_analysis ? {
              emotion: analysisData.title_analysis.emotion,
              emotionScore: analysisData.title_analysis.emotion_score,
              distortions: analysisData.title_analysis.distortions || [],
            } : null,
            overallSentiment: analysisData.overall_sentiment,
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
    
    const { userId } = auth();
    if (!userId) {
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
          
          updateData.analysis = {
            contentAnalysis: {
              emotion: analysisData.content_analysis?.emotion,
              emotionScore: analysisData.content_analysis?.emotion_score,
              intent: analysisData.content_analysis?.intent,
              intentScore: analysisData.content_analysis?.intent_score,
              risk: analysisData.content_analysis?.risk,
              riskScore: analysisData.content_analysis?.risk_score,
              distortions: analysisData.content_analysis?.distortions || [],
              distortionDetails: analysisData.content_analysis?.distortion_details?.map(d => ({
                distortionType: d.distortion_type,
                confidence: d.confidence,
                emoji: d.emoji,
                explanation: d.explanation,
                reframingSuggestion: d.reframing_suggestion
              })) || [],
              reframes: analysisData.content_analysis?.reframes || [],
              behavioralSuggestions: analysisData.content_analysis?.behavioral_suggestions || [],
              clinicianNotes: analysisData.content_analysis?.clinician_notes || [],
            },
            titleAnalysis: analysisData.title_analysis ? {
              emotion: analysisData.title_analysis.emotion,
              emotionScore: analysisData.title_analysis.emotion_score,
              distortions: analysisData.title_analysis.distortions || [],
            } : null,
            overallSentiment: analysisData.overall_sentiment,
            keyThemes: analysisData.key_themes || [],
            therapeuticInsights: analysisData.therapeutic_insights || [],
            progressIndicators: analysisData.progress_indicators || [],
            recommendations: analysisData.recommendations || [],
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
    
    const { userId } = auth();
    if (!userId) {
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
