import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import mongoose from 'mongoose';

// Mood Schema
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
  date: { type: String, required: true, index: true }, // YYYY-MM-DD format
  timestamp: { type: Date, default: Date.now },
}, {
  timestamps: true
});

// Create compound index for efficient querying
MoodSchema.index({ userId: 1, date: 1 });

const Mood = mongoose.models.Mood || mongoose.model('Mood', MoodSchema);

export async function POST(req) {
  try {
    const { userId } = await auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const body = await req.json();
    const { mood, intensity, notes } = body;

    // Validate required fields
    if (!mood || !intensity) {
      return NextResponse.json({ 
        error: 'Mood and intensity are required' 
      }, { status: 400 });
    }

    // Get today's date
    const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD

    // Check if user already logged mood today
    const existingMood = await Mood.findOne({
      userId,
      date: today
    });

    if (existingMood) {
      // Update existing mood for today
      existingMood.mood = mood;
      existingMood.intensity = intensity;
      existingMood.notes = notes || '';
      existingMood.timestamp = new Date();
      
      await existingMood.save();
      
      return NextResponse.json({
        message: 'Mood updated successfully',
        mood: existingMood
      });
    } else {
      // Create new mood entry
      const newMood = new Mood({
        userId,
        mood,
        intensity,
        notes: notes || '',
        date: today,
        timestamp: new Date()
      });

      await newMood.save();
      
      return NextResponse.json({
        message: 'Mood logged successfully',
        mood: newMood
      });
    }

  } catch (error) {
    console.error('Error in mood POST:', error);
    return NextResponse.json({ 
      error: 'Internal server error' 
    }, { status: 500 });
  }
}

export async function GET(req) {
  try {
    const { userId } = await auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { searchParams } = new URL(req.url);
    const requestedUserId = searchParams.get('userId');
    const limit = parseInt(searchParams.get('limit')) || 10;

    // Only allow users to access their own mood data
    const targetUserId = requestedUserId || userId;
    if (targetUserId !== userId) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Get recent moods
    const moods = await Mood.find({ userId: targetUserId })
      .sort({ timestamp: -1 })
      .limit(limit);

    return NextResponse.json(moods);

  } catch (error) {
    console.error('Error in mood GET:', error);
    return NextResponse.json({ 
      error: 'Internal server error' 
    }, { status: 500 });
  }
}