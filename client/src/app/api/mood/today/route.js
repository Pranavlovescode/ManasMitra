import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import mongoose from 'mongoose';

// Use the same Mood model from the main mood route
const MoodSchema = new mongoose.Schema({
  userId: { type: String, required: true, index: true },
  mood: { 
    type: String, 
    required: true,
    enum: ['sad', 'neutral', 'happy', 'excited', 'loved']
  },
  intensity: { 
    type: Number, 
    required: true, 
    min: 1, 
    max: 10 
  },
  notes: { type: String, default: '' },
  date: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
}, {
  timestamps: true
});

MoodSchema.index({ userId: 1, date: 1 });
const Mood = mongoose.models.Mood || mongoose.model('Mood', MoodSchema);

export async function GET(req) {
  try {
    const { userId } = await auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { searchParams } = new URL(req.url);
    const requestedUserId = searchParams.get('userId');
    const date = searchParams.get('date') || new Date().toISOString().split('T')[0];

    // Only allow users to access their own mood data
    const targetUserId = requestedUserId || userId;
    if (targetUserId !== userId) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Check if user has logged mood for the specified date (default: today)
    const moodEntry = await Mood.findOne({
      userId: targetUserId,
      date: date
    });

    return NextResponse.json({
      hasMoodToday: !!moodEntry,
      mood: moodEntry || null,
      date: date
    });

  } catch (error) {
    console.error('Error checking today mood:', error);
    return NextResponse.json({ 
      error: 'Internal server error' 
    }, { status: 500 });
  }
}